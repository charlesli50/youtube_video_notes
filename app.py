from flask import Flask, request, render_template, send_file, redirect, url_for
import os
from make_pdf import pipeline

app = Flask(__name__)
RESULT_FOLDER = 'outputs'
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    youtube_url = request.form['youtube_url']
    pdf_name = request.form['pdf_name']

    try:
        file_title = pdf_name.replace(' ', '_')
        pipeline(youtube_url, pdf_name, file_title)

        pdf_path = os.path.join(RESULT_FOLDER, pdf_name, f"{file_title}/{file_title}.pdf")
        print(pdf_path)
        if os.path.exists(pdf_path):
            return redirect(url_for('download', pdf_name=pdf_name, file_title=file_title))
        else:
            return "PDF generation failed. Please check the pipeline."

    except Exception as e:
        return f"Error processing the request: {e}"

@app.route('/download/<pdf_name>/<file_title>')
def download(pdf_name, file_title):
    pdf_path = os.path.join(RESULT_FOLDER, pdf_name, f"{file_title}/{file_title}.pdf")
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
