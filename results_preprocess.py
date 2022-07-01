import results_analysis as ra
import pickle


LABELS = ['greedy', 'random', 'UCB', 'EI', 'PI']

NOISE_LEVELS = ['0.0', '0.05', '0.1', '0.15000000000000002','0.2', '0.25']

def pro_folder(target, output):
    for noise in NOISE_LEVELS:
        data = ra.load_cumulative_data(target+'AL_noise', noise, 1, LABELS)
        file = open(output+f'/LCD{noise}.pkl', 'wb')
        pickle.dump(data,file)
        data2 = ra.load_true_cumulative_data(target+'AL_noise', noise, 1, LABELS)
        file = open(output+f'/LTCD{noise}.pkl', 'wb')
        pickle.dump(data2, file)

"""
def run(setn, batchs):
    target_folder = f'results_PubChem/set{setn}/{batchs}/' 
    output_folder = f'results_PubChem_C/set{setn}/{batchs}/'

    pro_folder(target_folder+'noR/', output_folder+'noR')
    pro_folder(target_folder+'wR/', output_folder+'wR')

sets = [1,2]
batchss = [100,300,500,1000]

for s in sets:
    for b in batchss:
        run(s, b)
"""
FNAME = 'friedman'

target = f'results_simulated/{FNAME}/'
output = f'results_simulated_C/{FNAME}/'

pro_folder(target+'noR/', output+'noR')
pro_folder(target+'wR/', output+'wR')
