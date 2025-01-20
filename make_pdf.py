
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import numpy as np
from pytubefix import YouTube
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
from dotenv import load_dotenv
import re
import os
import subprocess

from openai import OpenAI

load_dotenv()

client = OpenAI()
api_key = os.getenv('API_KEY')
client.api_key = api_key

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
os.environ["FFMPEG_BINARY"] = "/opt/homebrew/bin/ffmpeg"
print("FFmpeg Path:", os.environ["IMAGEIO_FFMPEG_EXE"])


def download_video(youtube_url, save_title, save_path="downloads/"):
    try:
        yt = YouTube(youtube_url)

        stream = yt.streams.get_highest_resolution()

        print(f"Downloading: {yt.title}")
        stream.download(output_path=save_path, filename=save_title)
        print(f"Downloaded successfully to {save_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")


def extract_keyframes(video_path, output_dir="output/selectedframes", num_keyframes=3, ffmpeg_path="/opt/homebrew/bin/ffmpeg"):
    try:
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        os.environ["FFMPEG_BINARY"] = ffmpeg_path
        print("FFmpeg Path:", os.environ["IMAGEIO_FFMPEG_EXE"])

        vd = Video()
        os.makedirs(output_dir, exist_ok=True)

        diskwriter = KeyFrameDiskWriter(location=output_dir)

        print(f"Input video file path = {video_path}")
        print(f"Output keyframes will be saved to: {output_dir}")

        vd.extract_video_keyframes(
            no_of_frames=num_keyframes, file_path=video_path, writer=diskwriter
        )

        print(f"Keyframes extracted and saved to {output_dir}")
    except Exception as e:
        print(f"Error during keyframe extraction: {e}")


def get_transcript(link):
#     https://www.youtube.com/watch?v=nt63k3bfXS0&list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU&index=6
    v = link.find("v=")+2
    video_id = link[v:v+11]
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return transcript


def preprocess_for_openai(transcript):
    combined_text = ""
    for segment in transcript:
        start_time = segment['start'] / 60
        combined_text += f"[{start_time:.2f}] {segment['text']}\n"
    return combined_text


def summarize_transcript(text_to_summarize, model="gpt-4o-mini"):
    
    prompt = f"""
    You are an expert at creating detailed, structured lecture notes from transcripts.

    Your task is to convert the provided lecture transcript into comprehensive, well-organized, and insightful lecture notes, as well as provide timestamps that align with the video's original transcript.

    TeX syntax should be utilized to explain and elaborate on the concepts within each subsection. Generate TeX inline math environments as necessary. 
    
    Each set of notes should follow this format:
    
    # Lecture Notes on [Topic Name]
    
    ## Introduction
    - Begin with a summary of the overall lecture and its goals.
    
    ## [Main Section Title]
    ### Subsection 1
    - Elaborate on the main point(s) of the subsection.
    
    ### Subsection 2
    - Continue explaining and expanding.

    ### Additional Analysis
    - Expand on the original content, and write a paragraph or two that integrate your own expertise and insights. This should provide an extensive, educational resource on the subject.
    
    ## [Next Main Section Title]
    - Organize subsequent sections similarly, ensuring a logical flow.
    
    ## Conclusion
    - Summarize the lecture's key takeaways.
    - Provide general advice or actionable insights related to the topic.
    
    ### General Tips
    - Include any advice or practical steps relevant to understanding or applying the concepts.

    Here is the lecture transcript:
    
    {text_to_summarize}

    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": prompt}
        ],
        seed=0,
        temperature=0.3
    )
    return completion



def save_summary_to_text(summarized, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(summarized)


def sanitize_for_latex(text):
    """
    Escapes special LaTeX characters in the given text,
    except for braces `{}` used in LaTeX commands.
    """
    special_chars = {
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "&": r"\&",
        "$": r"\$",
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text


def save_summary_to_latex(input_text, output_file):
    """
    Converts structured text to a LaTeX file.

    Args:
        input_text (str): The structured text input.
        output_file (str): The name of the output .tex file.
    """
    # LaTeX preamble and document start
    preamble = r"""
\documentclass{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{geometry}
\usepackage{graphicx}
\geometry{margin=1in}
\usepackage{hyperref}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
}

\title{Lecture Notes}
\author{}
\date{}

\begin{document}
\maketitle
"""
    end = r"\end{document}"

    # convert latex headers
    converted_text = re.sub(r"^# (.+)$", r"\\section*{\1}", input_text, flags=re.MULTILINE)
    converted_text = re.sub(r"^## (.+)$", r"\\subsection*{\1}", converted_text, flags=re.MULTILINE)
    converted_text = re.sub(r"^### (.+)$", r"\\subsubsection*{\1}", converted_text, flags=re.MULTILINE)

    converted_text = sanitize_for_latex(converted_text)

    # handle itemize
    def wrap_itemize(match):
        items = match.group(1).strip().split("\n")
        formatted_items = "\n".join([f"\\item {item[2:].strip()}" for item in items if item.strip()])
        return f"\\begin{{itemize}}\n{formatted_items}\n\\end{{itemize}}"
    converted_text = re.sub(r"(?m)(^- .+(?:\n- .+)*)", wrap_itemize, converted_text)

	#handle math environment
    converted_text = re.sub(r"\\\[([^\\]+)\\\]", r"\\[\1\\]", converted_text)
    converted_text = re.sub(r"\\\((.+?)\\\)", r"$\1$", converted_text)

    converted_text = re.sub(r"([a-zA-Z])_([a-zA-Z0-9]+)", r"{\1_\{\2\}}", converted_text)
    converted_text = re.sub(r"([a-zA-Z])\^([a-zA-Z0-9]+)", r"{\1^\{\2\}}", converted_text)

    latex_content = f"{preamble}{converted_text}\n{end}"
    # print(latex_content)

    with open(output_file, 'w') as tex_file:
        tex_file.write(latex_content)

    print(f"LaTeX file '{output_file}' generated successfully.")


def tex_to_pdf(tex_file, output_dir=None, keyframe_paths=[]):
    if not tex_file.endswith(".tex"):
        raise ValueError("Input file must be a .tex file")
    
    if output_dir is None:
        output_dir = os.path.dirname(tex_file) or "."
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Modify the LaTeX content to include keyframe images at the top
    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Generate the keyframe LaTeX block
    keyframes_tex = "\n".join(
        f"\\includegraphics[width=0.3\\textwidth]{{{path}}}\\hspace{{5mm}}"
        for path in keyframe_paths
    )
    keyframes_tex = f"\\begin{{center}}\n{keyframes_tex}\n\\end{{center}}\n"

    # insert keyframes
    section_marker = r"\section*{"
    insert_position = content.find(section_marker)
    if insert_position != -1:
        section_end = content.find("}", insert_position) + 1
        content = content[:section_end] + "\n" + keyframes_tex + content[section_end:]
    else:
        print("Warning: Section header not found. Keyframes will not be inserted under a section.")
    
    # Save the modified LaTeX file
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(content)

    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-file-line-error", "-output-directory", output_dir, tex_file],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        pdf_file = os.path.join(output_dir, os.path.basename(tex_file).replace(".tex", ".pdf"))
        if os.path.exists(pdf_file):
            print(f"PDF generated: {pdf_file}")
            return pdf_file
        else:
            raise FileNotFoundError("PDF generation failed. Check the LaTeX log for errors.")
    except subprocess.CalledProcessError as e:
        print("Error during LaTeX compilation:")
        print(e.stdout) 
        raise


def pipeline(youtube_url, pdf_name, file_title, num_keyframes=3, save_title="video.mp4", ffmpeg_path="/opt/homebrew/bin/ffmpeg"):
    try:
        # Define output directories
        base_output_dir = f"outputs/{pdf_name}"
        video_output_dir = f"{base_output_dir}/downloads"
        keyframes_output_dir = f"{base_output_dir}/keyframes"
        text_output_dir = f"{base_output_dir}/texts"
        
        # Ensure directories exist
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(keyframes_output_dir, exist_ok=True)
        os.makedirs(text_output_dir, exist_ok=True)

        # Step 1: Download video
        save_path = os.path.join(video_output_dir, save_title)
        print("Downloading video...")
        download_video(youtube_url, save_title, save_path=video_output_dir)

        # Step 2: Extract keyframes
        print("Extracting keyframes...")
        extract_keyframes(save_path, output_dir=keyframes_output_dir, num_keyframes=num_keyframes, ffmpeg_path=ffmpeg_path)

        # Step 3: Fetch and process transcript
        print("Fetching transcript...")
        transcript = get_transcript(youtube_url)
        print("Processing transcript...")
        processed_transcript = preprocess_for_openai(transcript)

        # Step 4: Summarize transcript
        print("Summarizing transcript...")
        summarized = summarize_transcript(processed_transcript)
        summarized = summarized.choices[0].message.content

        # Step 5: Save summary to text
        text_file_path = os.path.join(text_output_dir, f"{file_title}.txt")
        print("Saving summary to text file...")
        save_summary_to_text(summarized, text_file_path)

        # Step 6: Convert summary to LaTeX
        tex_file_path = os.path.join(base_output_dir, f"{file_title}.tex")
        print("Converting summary to LaTeX...")
        save_summary_to_latex(summarized, tex_file_path)

        # Step 7: Prepare keyframe paths
        keyframe_paths = [
            os.path.join(keyframes_output_dir, filename)
            for filename in sorted(os.listdir(keyframes_output_dir))
            if filename.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        print(f"Found keyframes: {keyframe_paths}")

        # Step 8: Convert LaTeX to PDF
        print("Converting LaTeX to PDF...")
        tex_to_pdf(tex_file_path, os.path.join(base_output_dir, file_title), keyframe_paths = keyframe_paths)

        print("Pipeline completed successfully! All outputs saved in:", base_output_dir)

    except Exception as e:
        print(f"Error in pipeline: {e}")
