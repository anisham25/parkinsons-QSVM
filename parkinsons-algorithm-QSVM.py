#Import necessary packages
import numpy as np
import pandas as pd
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.components.feature_maps import SecondOrderExpansion
from qiskit.aqua.components.multiclass_extensions import (ErrorCorrectingCode,AllPairs,OneAgainstRest)
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import get_feature_dimension
from qiskit import IBMQ
from sklearn.model_selection import train_test_split


#Load account credentials
provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
#Connect to quantum simulator
backend = provider.get_backend('ibmq_qasm_simulator')


#Ignore all deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


#Number of times the jobs will be run on the quantum device
shots = 500 


#Load dataset
p_now = [[1,0.85247,0.71826,0.57227,240,239,0.008064,0.000087,0.00218],[1,0.76686,0.69481,0.53966,234,233,0.008258,0.000073,0.00195],
[1,0.85083,0.67604,0.58982,232,231,0.008340,0.000060,0.00176],
[0,0.41121,0.79672,0.59257,178,177,0.010858,0.000183,0.00419],
[0,0.32790,0.79782,0.53028,236,235,0.008162,0.002669,0.00535]]
notp =[[1, 0.84881, 0.60125, 0.44782, 351, 350, 0.005510131, 7.27E-05, 0.00165],
[1, 0.70649, 0.60081, 0.50228, 339, 338, 0.005697116, 8.77E-05, 0.00174],
[1, 0.84703, 0.62323, 0.45198, 333, 332, 0.00579286, 5.74E-05, 0.00107],
[1, 0.78601, 0.61539, 0.20724, 594, 593, 0.003249218, 3.45E-05, 0.00039],
[1, 0.78213, 0.61626, 0.17873, 598, 597, 0.003226095, 2.5E-05, 0.00037]]


#Divide dataset into training and testing with a 7:3 ratio
p_train, p_test, notp_train, notp_test = train_test_split(p_now, notp, test_size=0.3, random_state=12)


#Combine training/testing arrays
training_data={'A': p_train, 'B': notp_train}
testing_data = {'A' : p_test, 'B': notp_test}


num_qubits=9


feature_map = SecondOrderExpansion(feature_dimension=num_qubits, depth = 2, entanglement = 'full')


svm = QSVM(feature_map,training_data,testing_data)


quantum_instance=QuantumInstance(backend, shots=shots, skip_qobj_validation=False)


#Run the QSVM for accuracy results
result = svm.run(quantum_instance)


#Unlabelled data
data = np.array([[0, 0.80766, 0.73961, 0.20569, 445, 444, 0.004334704, 2.43E-05, 0.00072],
[1, 0.83967, 0.80944, 0.45038, 259, 257, 0.007310325, 0.00251383, 0.00534],
[1, 0.81525, 0.73462, 0.64849, 303, 302, 0.006381791, 0.000149359, 0.00292],
[1, 0.79163, 0.80358, 0.43866, 330, 329, 0.005844644, 8.21E-05, 0.00278],
[0, 0.7594, 0.68265, 0.39428, 443, 442, 0.004354498, 6.69E-05, 0.00076],
[1, 0.81547, 0.64809, 0.60227, 243, 242, 0.007963248, 9.04E-05, 0.00294],
[1, 0.82586, 0.59259, 0.44395, 354, 353, 0.005455777, 4.0E-05, 0.00129],
[1, 0.73403, 0.61812, 0.50801, 343, 342, 0.00563253, 8.34E-05, 0.00167],
[1, 0.87601, 0.62297, 0.43552, 346, 345, 0.005573364, 5.66E-05, 0.00118]
])


#Test unlabelled data
prediction = svm.predict(data,quantum_instance)


print('Prediction of Parkinsons disease based upon speech indicators\n')
print('Accuracy: ' , result['testing_accuracy'],'\n')
print('Prediction from input data where 0 = Healthy and 1 = Present\n')
print(prediction)
