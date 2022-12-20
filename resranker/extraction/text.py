import os
import fitz
import docx2txt


class Pdf2Txt:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_text = ""

    def _check_pdf(self):
        return self.file_path.endswith(".pdf")

    def _check_exists(self):
        return os.path.isfile(self.file_path)

    def is_valid(self):
        return True if (self._check_pdf() and self._check_exists()) else False

    def extract(self):
        with fitz.open(self.file_path) as pdf:
            for page in pdf:
                self.raw_text += page.get_text()
        return self.raw_text


class Doc2Txt:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_text = ""

    def _check_docx(self):
        return self.file_path.endswith(".docx")

    def _check_exists(self):
        return os.path.isfile(self.file_path)

    def is_valid(self):
        return True if (self._check_docx() and self._check_exists()) else False

    def extract(self):
        self.raw_text = docx2txt.process(self.file_path)
        return self.raw_text
