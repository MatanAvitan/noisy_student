import os
from consts import TRAIN_SPLIT_PERCENTAGES, STUDENT_TEACHER_LOOP, ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR, ORIGINAL_ANNOTATIONS_DIR, ORIGINAL_TRAIN_ANNOTATION_FILE

split_command = """python3 cocosplit.py \
                  --having-annotations \
                  -s {train_split_percentage} \
                  {annotation_file_path} \
                  {train_output_file} \
                  {test_output_file}"""

class DataSplitter():
    def __init__(self, annotation_file_path, train_split_percentage, train_output_file, test_output_file):
        self.annotation_file_path = annotation_file_path
        self.train_split_percentage = train_split_percentage
        self.train_output_file = train_output_file
        self.test_output_file = test_output_file

    def split(self):
        os.system(split_command.format(train_split_percentage=self.train_split_percentage,
                                                     annotation_file_path=self.annotation_file_path,
                                                     train_output_file=self.train_output_file,
                                                     test_output_file=self.test_output_file))
def main():
    initial_model_idx = 0
    train_split_percentage = TRAIN_SPLIT_PERCENTAGES[0]
    splitter = DataSplitter(annotation_file_path=os.path.join(ANNOTATIONS_DIR,ORIGINAL_ANNOTATIONS_DIR, ORIGINAL_TRAIN_ANNOTATION_FILE),
                            train_split_percentage=train_split_percentage,
                            train_output_file=os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'annotations_file_model_idx_{model_idx}'.format(model_idx=initial_model_idx)),
                            test_output_file=os.path.join(ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR, ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'unused_annotations_file_model_idx_{model_idx}'.format(model_idx=initial_model_idx)))

    splitter.split()
    for curr_model_idx, train_split_percentage in zip(range(initial_model_idx+1,STUDENT_TEACHER_LOOP), TRAIN_SPLIT_PERCENTAGES[1:]):
        annotation_file_path = os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'unused_annotations_file_model_idx_{prev_mode_idx}'.format(prev_mode_idx=curr_model_idx-1))
        train_output_file = os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'annotations_file_model_idx_{curr_model_idx}'.format(curr_model_idx=curr_model_idx))
        if curr_model_idx == STUDENT_TEACHER_LOOP-1:
            test_output_file = os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'annotations_file_model_idx_{curr_model_idx}'.format(curr_model_idx=curr_model_idx+1))
        else:
            test_output_file = os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'unused_annotations_file_model_idx_{curr_model_idx}'.format(curr_model_idx=curr_model_idx))
        splitter = DataSplitter(annotation_file_path=annotation_file_path,
                                train_split_percentage=train_split_percentage,
                                train_output_file=os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'annotations_file_model_idx_{curr_model_idx}'.format(curr_model_idx=curr_model_idx)),
                                test_output_file=test_output_file)
        splitter.split()
    # rename last unused_annotations_file to annotations_file
    for model_idx in range(0,STUDENT_TEACHER_LOOP-1):
       os.remove(os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'unused_annotations_file_model_idx_{model_idx}'.format(model_idx=model_idx)))
if __name__ == '__main__':
    main()
