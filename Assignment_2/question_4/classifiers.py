from coherence_scorer import get_coherence_model, get_coherence_score
from grammaticality_scorer import get_grammaticallity_model, get_grammaticallity_score
from non_reduncancy_scorer import get_non_redundancy_model, get_non_redundancy_score
from utils import read_summarries, read_test_data, read_train_data


if __name__ == "__main__":
    # Get Train Data
    train_coherence, train_grammaticality, train_nonredundancy = read_train_data()
    test_coherence, test_grammaticality, test_nonredundancy = read_test_data()
    train_files = list(train_coherence.keys())
    test_files = list(test_coherence.keys())
    summarries = read_summarries()
    # Train Classifiers
    gramaticallity_model = get_grammaticallity_model(train_files, summarries, train_grammaticality)
    non_redundancy_model = get_non_redundancy_model(train_files, summarries, train_nonredundancy)
    coherence_model = get_coherence_model(train_files, summarries, train_coherence)
    # Test Classifiers
    MSE, pearson_cor = get_grammaticallity_score(
        gramaticallity_model, test_files, summarries, test_grammaticality
    )
    print(MSE, pearson_cor)
    MSE, pearson_cor = get_non_redundancy_score(
        non_redundancy_model, test_files, summarries, test_nonredundancy
    )
    print(MSE, pearson_cor)
    MSE, pearson_cor = get_coherence_score(
        coherence_model, test_files, summarries, test_coherence
    )
    print(MSE, pearson_cor)
