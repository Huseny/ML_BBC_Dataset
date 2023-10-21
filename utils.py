# .classes : contains label for each document
# .docs : identifies the catagory of the document and it's identifier. eg: business.004 shows
#         a document with catagory business and identifier 004
# .mtx : shows the frequency of each word. eg: 1 21 1.0 shows "1" represents the term number in
#        bbc.terms file, "21" represents corresponds to a documentId in the bbc.classes file
#        "1.0" represents the frequency
import numpy as np
from collections import defaultdict
from decimal import Decimal, getcontext

getcontext().prec = 20


class Utils:
    def __init__(self) -> None:
        pass

    def extract_raw_data(self) -> list:
        raw_data = []
        with open("dataset/bbc.classes") as file:
            for line_num, line in enumerate(file):
                if line_num > 3:
                    document_id, class_id = line.rstrip().split()
                    raw_data.append([document_id, class_id])
        return raw_data

    def extract_terms_raw_data(self) -> list:
        terms_raw_data = []
        with open("dataset/bbc.mtx", newline="") as mtx:
            for line_num, line in enumerate(mtx):
                if line_num >= 2:
                    term_id, document_id, freq = line.rstrip().split()
                    terms_raw_data.append([term_id, document_id, freq])

        return terms_raw_data

    def get_classes_prob(self, raw_data: list) -> dict:
        classes_count = defaultdict(int)
        total_data = len(raw_data)

        for document_id, class_id in raw_data:
            classes_count[class_id] += 1

        return {
            class_id: Decimal(class_count) / Decimal(total_data)
            for class_id, class_count in classes_count.items()
        }

    def beautify_data(self, raw_data: list) -> dict:
        beautified_data = {}
        for document_id, class_id in raw_data:
            beautified_data[document_id] = class_id

        return beautified_data

    def get_terms_probability(self, class_terms: dict) -> tuple[dict, dict]:
        class_probabilities = {}
        total_terms = {}
        for class_id, term_freqs in class_terms.items():
            total_freq = sum(term_freqs.values())
            total_terms[class_id] = total_freq
            term_probabilities = {
                term_id: Decimal(freq) / Decimal(total_freq)
                for term_id, freq in term_freqs.items()
            }
            class_probabilities[class_id] = term_probabilities

        return class_probabilities, total_terms

    def get_document_terms(self) -> dict:
        # store the terms of each document in the form {document_id: [terms]}
        document_terms: dict = {}

        with open("dataset/bbc.mtx", newline="") as mtx:
            for line_num, line in enumerate(mtx):
                if line_num >= 2:
                    term_id, document_id, freq = (val for val in line.split())
                    document_dict = document_terms.setdefault(
                        str(int(document_id) - 1), {}
                    )
                    document_dict[term_id] = freq

        return document_terms

    def get_logistic_data(
        self, documents_data: dict, document_terms: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        # document_terms = {document_id: {termid: freq}}
        # documents_data = {document_id: label}
        sorted_documents_data = dict(
            sorted(documents_data.items(), key=lambda x: int(x[0]))
        )

        with open("dataset/bbc.terms", "r") as file:
            total_terms = sum(1 for line in file)

        x = defaultdict(int)
        y = [int(label) for label in sorted_documents_data.values()]
        y = np.array(y)
        for document_id in sorted_documents_data:
            terms: dict = document_terms[document_id]
            x_part = x.setdefault(document_id, [])
            for termid in range(1, total_terms + 1):
                if str(termid) in terms.keys():
                    x_part.append(int(float(terms[str(termid)])))
                else:
                    x_part.append(0)
        real_x = [real_value for real_value in x.values()]

        return np.array(real_x), y
