import numpy as np
from utils import Utils
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression
import random
from collections import defaultdict


class Experiment:
    def __init__(self) -> None:
        self.utils = Utils()
        self.training_data, self.test_data = self.split_dataset()

        self.prior_prob = self.utils.get_classes_prob(self.training_data)
        self.training_documents_data = self.utils.beautify_data(self.training_data)

        self.test_documents_data = self.utils.beautify_data(self.test_data)

        self.training_terms_data = self.split_terms_dataset(
            self.training_documents_data
        )
        self.terms_prior_prob, self.total_terms = self.utils.get_terms_probability(
            self.training_terms_data
        )

        self.document_terms = self.utils.get_document_terms()

        self.training_x, self.training_y = self.utils.get_logistic_data(
            self.training_documents_data, self.document_terms
        )

        self.test_x, self.test_y = self.utils.get_logistic_data(
            self.test_documents_data, self.document_terms
        )

    def do_naive_bayes(self, laplace_smoothing: float) -> float:
        naive_bayes = NaiveBayes(
            self.prior_prob,
            self.terms_prior_prob,
            self.document_terms,
            self.total_terms,
        )
        predicted_labels: dict = naive_bayes.get_dic_classification(
            self.test_documents_data, laplace_smoothing
        )

        return self.calculate_accuracy_naive(predicted_labels, self.test_documents_data)

    def do_logistic_regression(
        self, num_iter: int = 10, learning_rate: float = 0.01
    ) -> float:
        logistic_regression = LogisticRegression()
        weights = logistic_regression.learn(
            self.training_x, self.training_y, learning_rate, num_iter
        )

        predicted_label = logistic_regression.classify(self.test_x, weights)

        return self.calculate_accuracy_logistic(predicted_label, self.test_y)

    def split_dataset(self, test_data_ratio: float = 0.2) -> tuple[list, list]:
        raw_data: list = self.utils.extract_raw_data()

        split_index = int(round(test_data_ratio * len(raw_data)))

        random.shuffle(raw_data)

        test_data = raw_data[:split_index]
        training_data = raw_data[split_index:]

        return training_data, test_data

    def split_terms_dataset(self, training_documents_data: dict) -> dict:
        raw_terms_data: list = self.utils.extract_terms_raw_data()
        class_terms = defaultdict(lambda: defaultdict(int))

        for term_id, document_id, freq in raw_terms_data:
            if document_id in training_documents_data:
                class_id = training_documents_data[document_id]
                class_terms[class_id][term_id] += float(freq)

        return class_terms

    def calculate_accuracy_naive(
        self, predicted_labels: dict, original_labels: dict
    ) -> float:
        correct_count = 0
        total_count = len(predicted_labels)

        for document_id, predicted_label in predicted_labels.items():
            original_label = original_labels[document_id]
            if predicted_label == original_label:
                correct_count += 1

        accuracy = correct_count / total_count
        return accuracy * 100

    def calculate_accuracy_logistic(self, y_pred, y_true):
        correct_predictions = np.sum(y_true == y_pred)
        total_examples = len(y_true)
        accuracy = correct_predictions / total_examples
        return accuracy * 100
