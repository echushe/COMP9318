import submission as submission
import fake_target_machine as target

test_data='./test_data.txt'
strategy_instance = submission.fool_classifier(test_data)

target.train_target_and_validate_changed_txt('./test_data.txt', './modified_data.txt')
