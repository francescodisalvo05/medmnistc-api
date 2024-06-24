from medmnistc.utils.baselines import BASELINES

from sklearn.metrics import balanced_accuracy_score
import numpy as np
import json
import os


class Evaluator:
    def __init__(self, 
                 dataset_name: str, 
                 true_labels: list, 
                 corruption_types: list, 
                 output_folder: str, 
                 architecture: str, 
                 task: str,
                 suffix_log: str = ''):
        """
        Evaluates the robustness of a given model on a set of pre-defined corruptions.

        :param dataset_name: Name of the dataset (used for logging).
        :param true_labels: True labels of the current dataset.
        :param corruption_types: List of corruptions used for the current experiment.
        :param output_folder: Where to store the output logs (json file).
        :param architecture: Name of the architecture (logging purposes).
        :param task: Classification task (i.e., binary-class etc).
        :param suffix: Suffix of the logging file (e.g. seed of the current experiment)
        """
        self.dataset_name = dataset_name
        self.len_dataset = len(true_labels)
        self.corruption_types = corruption_types
        self.output_folder = output_folder
        self.architecture = architecture
        self.task = task
        self.suffix_log = suffix_log
        
        self.initialize_evaluation()

        self.true_labels = np.array(true_labels)
        if self.true_labels.shape[1] == 1: # multi-class or binary
            self.true_labels = self.true_labels.reshape(-1,) # flatten


    def initialize_evaluation(self):
        """
        Load the baseline logging file, if required, and setup evaluation function based
        on the current classification task.
        """
        self.corruption_errors = {corruption: [] for corruption in self.corruption_types}
        self.clean_score = None  # Init

        assert self.dataset_name in BASELINES.keys(), f"{self.dataset_name} has no pre-defined baselines in /utils/baselines.py"
        self.corruption_errors_alexnet = BASELINES[self.dataset_name]
        
        self.evaluation_metric = self.get_eval_metric()


    def get_eval_metric(self):
        """
        Define the appropriate evaluation function based on the current task.
        """
        # Return the average balanced accuracy per label, using the chosen operating points (per label).
        if self.task == "multi-label, binary-class":
            return lambda y_true, y_score, threshold: 1.0 - np.mean(
                [balanced_accuracy_score(y_true[:, i], y_score[:, i] > threshold[i]) for i in range(y_true.shape[1])]
            )
        
        # Returns the balanced accuracy, using the chosen operating point.
        elif self.task == "binary-class":
            return lambda y_true, y_score, threshold: 1.0 - balanced_accuracy_score(y_true, y_score[:, -1] > threshold)
        
        # Returns the balanced accuracy, neglecting the default `threshold` argument.
        elif self.task == "multi-class" or self.task == "ordinal-regression":
            return lambda y_true, y_score, _: 1.0 - balanced_accuracy_score(y_true, np.argmax(y_score, axis=-1))

        else:
            raise ValueError(f"Unknown task type {self.task}")
        

    def evaluate(self, predicted_probabilities, corruption_type, threshold=0.5):
        """
        Evaluate the predictions of the current model.

        :param predicted_probabilities: List of raw predictions (i.e., probabilities)
                                        Note that the dataset is "repeatd" 5 times 
                                        due to the 5 increasing severities.                                    
        :param corruption_type: Name of the current corruption.
        :param threshold: Operating point(s) based on the given task.
                          float if task == "binary-class"
                          list[float] if task == "multi-label, binary-class"
                          None if task == "multi-class" or task == "ordinal-regression"
        """

        for severity in range(5):
            # get probabilities of the current severity slice
            index_range = slice(self.len_dataset * severity, self.len_dataset * (severity + 1))
            curr_prob = predicted_probabilities[index_range]
            # calculate relative score and update evaluation metric
            score = self.evaluation_metric(self.true_labels, curr_prob, threshold)
            self.corruption_errors[corruption_type].append(score)

    
    def evaluate_clean(self, predicted_probabilities, threshold=0.5):   
        """
        Evaluate clean dataset in order to calculate the relative corruption error.

        :param predicted_probabilities: List of raw predictions (i.e., probabilities)
                                        Note that the dataset is "repeatd" 5 times 
                                        due to the 5 increasing severities.
        :param threshold: Operating point(s) based on the given task.
                          float if task == "binary-class"
                          list[float] if task == "multi-label, binary-class"
                          None if task == "multi-class" or task == "ordinal-regression"
        """
        score = self.evaluation_metric(self.true_labels, predicted_probabilities, threshold)
        self.clean_score = score


    def dump_summary(self):
        """
        Store a json file containing the aggregated and raw results.
        """
        self.output_log = {}
        self.populate_summary()
        
        full_output_path = os.path.join(self.output_folder, f'{self.dataset_name}_{self.architecture}_{self.suffix_log}.json')
        with open(full_output_path, 'w') as f:
            json.dump(self.output_log, f, indent=4)

        print(f'Logs stored at `{full_output_path}`')


    def populate_summary(self):
        """
        Calculate the error metrics according to the formulas reported on the paper.
        """

        assert self.clean_score, "You first need to compute the clean error via self.evaluate_clean(...)"

        self.output_log['metrics'] = {'clean_score': self.clean_score or 0}
        self.output_log['be_scores'] = {}
        self.output_log['rbe_scores'] = {}

        for corruption, errors in self.corruption_errors.items():
            
            # For other architectures, normalize errors against AlexNet's performance
            alexnet_error = self.corruption_errors_alexnet['raw_scores'][corruption]
            alexnet_clean_score = self.corruption_errors_alexnet['clean_score']
            
            # Ensure there are scores to normalize against to avoid division by zero
            if alexnet_error and alexnet_clean_score is not None:
                # Normalized Balanced Error (BE)
                be = np.mean(errors) / alexnet_error # alexnet_error is already averaged across severities
                
                # Calculate Relative Balanced Error (RBE)
                rbe_num = np.mean(errors) - self.clean_score    # the clean score would be subtracted 5 times and divided by 5 (mean). So, we can put this out
                rbe_denom = alexnet_error - alexnet_clean_score # same here
                rbe = rbe_num / rbe_denom

            self.output_log['be_scores'][corruption] = be
            self.output_log['rbe_scores'][corruption] = rbe

        # Compute overall corrupted score and relative corrupted error
        self.output_log['metrics']['be'] = np.mean(list(self.output_log['be_scores'].values()))
        self.output_log['metrics']['rbe'] = np.mean(list(self.output_log['rbe_scores'].values()))
        
        # Include raw scores for completeness
        self.output_log['raw_scores'] = {k:np.mean(v) for k,v in self.corruption_errors.items()} # store only the average