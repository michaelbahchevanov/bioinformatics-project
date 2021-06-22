from lib.utils import projection
import numpy as np

class RMTClassifier(object):
    def __init__(self, threshold_multiple=1, train_cutoff=0.95):
        self.thresh_multiple = threshold_multiple
        self.train_cutoff = train_cutoff
        
        self.epsilon = None
        self.feature_vecs = None
        self.dim_V = None
        self.p = None
        
    def fit(self, train_mat):
        N, p = train_mat.shape
        self.p = p
        gamma = p/N
        thresh = ((1 + np.sqrt(gamma)) ** 2) * self.thresh_multiple
        
        C = np.dot(train_mat.T, train_mat)/N
        
        evals, evecs = np.linalg.eig(C)
        idx = evals.argsort()
        idx = idx[::-1]
        evals = evals[idx]
        evecs = evecs[: ,idx]
        
        dim_V = evals[evals > thresh].shape[0]
        feature_vecs = evecs[:, :dim_V]
        
        self.dim_V, self.feature_vecs = dim_V, feature_vecs
        
        similarity = []
        
        for mol_vec_idx in range(N):
            trial_molecule = train_mat[mol_vec_idx, :]
            trial_molecule_proj = projection(trial_molecule, dim_V, p, feature_vecs)
            similarity.append(np.linalg.norm(trial_molecule - trial_molecule_proj))
        
        similarity.sort()
        similarity = np.array(similarity)
        cutoff_idx = int(self.train_cutoff * len(similarity))
        epsilon = similarity[cutoff_idx]
        self.epsilon = epsilon
        
    def predict(self, test_set, epsilon_multiple=1):
        test_set_similarity = []
        for mol_vec_idx in range(test_set.shape[0]):
            trial_molecule = test_set[mol_vec_idx, :]
            trial_molecule_proj = projection(trial_molecule, self.dim_V, self.p, self.feature_vecs)
            test_set_similarity.append(np.linalg.norm(trial_molecule - trial_molecule_proj))
        
        predictions = np.array([1 if x < self.epsilon*epsilon_multiple else 0 for x in test_set_similarity])
        
        return predictions