from decimal import Decimal, getcontext

getcontext().prec = 20


class NaiveBayes:
    def __init__(
        self,
        classes_prob: dict,
        terms_prob: dict,
        document_terms: dict,
        total_terms: dict,
    ) -> None:
        self.classes_prob = classes_prob
        self.terms_prob = terms_prob
        self.document_terms = document_terms
        self.total_terms_dict = total_terms

    def get_dic_classification(self, data: dict, laplace_smoothing: float) -> dict:
        probabilities = {}
        for document_id in data:
            probabilities[document_id] = self.classifier(document_id, laplace_smoothing)

        return probabilities

    def classifier(self, document_id: str, laplace_smoothing: float):
        # p(class | document) = p(document | class) * p(class)
        # p(class | document) = p(term1 | class) * p(term2 | class) * ... * p(class)
        max_probability = float("-inf")
        max_class_label = None

        for class_label in self.classes_prob:
            total_terms = self.total_terms_dict[class_label]
            probability = 1.0
            for term in self.document_terms[document_id]:
                if term in self.terms_prob[class_label]:
                    probability = Decimal(probability) * Decimal(
                        self.terms_prob[class_label][term]
                    )
                else:
                    probability = Decimal(probability) * (
                        Decimal(laplace_smoothing)
                        / Decimal(total_terms + laplace_smoothing)
                    )

            probability = Decimal(probability) * Decimal(self.classes_prob[class_label])
            if probability > max_probability:
                max_probability = probability
                max_class_label = class_label

        return max_class_label
