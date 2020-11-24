import math
import logging

stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

class BM25:
    """
    Best Match 25.

    Parameters
    ----------
    k1 : float, default 1.5

    b : float, default 0.75

    Attributes
    ----------
    tf_ : list[dict[str, int]]
        Term Frequency per document. So [{'hi': 1}] means
        the first document contains the term 'hi' 1 time.

    df_ : dict[str, int]
        Document Frequency per term. i.e. Number of documents in the
        corpus that contains the term.

    idf_ : dict[str, float]
        Inverse Document Frequency per term.

    doc_len_ : list[int]
        Number of terms per document. So [3] means the first
        document contains 3 terms.

    corpus_ : list[list[str]]
        The input corpus.

    corpus_size_ : int
        Number of documents in the corpus.

    avg_doc_len_ : float
        Average number of terms for documents in the corpus.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1

    def fit(self, corpus):
        """
        Fit the various statistics that are required to calculate BM25 ranking
        score using the corpus given.

        Parameters
        ----------
        corpus : list[list[str]]
            Each element in the list represents a document, and each document
            is a list of the terms.

        Returns
        -------
        self
        """
        tf = []
        df = {}
        idf = {}
        doc_len = []
        corpus_size = 0
        for document in corpus:
            corpus_size += 1
            doc_len.append(len(document))

            # compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            tf.append(frequencies)

            # compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = df.get(term, 0) + 1
                df[term] = df_count

        for term, freq in df.items():
            idf[term] = math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))

        self.tf_ = tf
        self.df_ = df
        self.idf_ = idf
        self.doc_len_ = doc_len
        self.corpus_ = corpus
        self.corpus_size_ = corpus_size
        self.avg_doc_len_ = sum(doc_len) / corpus_size
        return self

    def search(self, query):
        scores = [self._score(query, index) for index in range(self.corpus_size_)]
        return scores

    def _score(self, query, index):
        score = 0.0

        doc_len = self.doc_len_[index]
        frequencies = self.tf_[index]
        for term in query:
            if term not in frequencies:
                continue

            freq = frequencies[term]
            numerator = self.idf_[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len_)
            score += (numerator / denominator)

        return score

def preprocess_text(corpus):
    ''' Prepare text for BM25-compatibility '''
    texts = [
        [word for word in document.lower().split() if word not in stopwords]
        for document in corpus
    ]

    # Build a word count dictionary so we can remove words that appear only once
    word_count_dict = {}
    for text in texts:
        for token in text:
            word_count = word_count_dict.get(token, 0) + 1
            word_count_dict[token] = word_count

    texts = [[token for token in text if word_count_dict[token] > 1] for text in texts]
    return texts

def main(query, corpus, sources, bm_n_doc):
    logging.warning('Applying BM25 algorithm ...')
    # Query our corpus to see which document is more relevant
    query = [word for word in query.lower().split() if word not in stopwords]
    # Preprocess text to remove stopwords and stuff
    texts = preprocess_text(corpus)
    # Fit and score texts on BM25
    bm25 = BM25()
    bm25.fit(texts)
    scores = bm25.search(query)
    # Sort by relevance score
    bm25_ranked = sorted(zip(scores, corpus, sources), key = lambda x: x[0], reverse=True)
    # Unpack the zipped lists
    scores, corpus, sources = zip(*bm25_ranked[:bm_n_doc])
    return list(corpus), list(sources)

if __name__ ==  '__main__':
    # Test question + context
    corpus = [
        'Human machine interface for lab abc computer applications',
        'A survey of user opinion of computer system response time',
        'The EPS user interface management system',
        'System and human system engineering testing of EPS',
        'Relation of user perceived response time to error measurement',
        'The generation of random binary unordered trees',
        'The intersection graph of paths in trees',
        'Graph minors IV Widths of trees and well quasi ordering',
        'Graph minors A survey'
    ]
    sources = [
        {'metadata_storage_name': 'bla',
        'document_id': 'bla',
        'document_uri': 'bla',
        'title': 'bla'},
        {'metadata_storage_name': 'bla',
        'document_id': 'bla',
        'document_uri': 'bla',
        'title': 'bla'},
        {'metadata_storage_name': 'bla',
        'document_id': 'bla',
        'document_uri': 'bla',
        'title': 'bla'},
        {'metadata_storage_name': 'bla',
        'document_id': 'bla',
        'document_uri': 'bla',
        'title': 'bla'},
        {'metadata_storage_name': 'bla',
        'document_id': 'bla',
        'document_uri': 'bla',
        'title': 'bla'}]
    main("The intersection of graph survey and trees", corpus, sources)