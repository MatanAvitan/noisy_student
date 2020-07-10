import os
from data_consts import (TRAIN_SPLIT_PERCENTAGE,
                         COCOSPLIT_PATH,
                         ANNOTATIONS_DIR,
                         NEW_ANNOTATIONS_DIR,
                         ORIGINAL_ANNOTATIONS_DIR,
                         ORIGINAL_TRAIN_ANNOTATION_FILE,
                         NEW_ANNOTATIONS_FILE_PREFIX,
                         ANNOTATIONS_FILE_FULL_MODEL)

split_command = """python {cocosplit_path} \
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
        os.system(split_command.format(cocosplit_path=COCOSPLIT_PATH,
                                       train_split_percentage=self.train_split_percentage,
                                       annotation_file_path=self.annotation_file_path,
                                       train_output_file=self.train_output_file,
                                       test_output_file=self.test_output_file))
def main():
    initial_model_idx = 0
    train_split_percentage = TRAIN_SPLIT_PERCENTAGE

    # split for 2 files: teacher initial data and the rest
    splitter = DataSplitter(annotation_file_path=os.path.join(ANNOTATIONS_DIR,ORIGINAL_ANNOTATIONS_DIR, ORIGINAL_TRAIN_ANNOTATION_FILE),
                            train_split_percentage=train_split_percentage,
                            train_output_file=os.path.join(ANNOTATIONS_DIR,NEW_ANNOTATIONS_DIR,'{prefix}_{model_idx}'.format(prefix=NEW_ANNOTATIONS_FILE_PREFIX,
                                                                                                                             model_idx=initial_model_idx)),
                            test_output_file=os.path.join(ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR,'{prefix}_{model_idx}'.format(prefix=NEW_ANNOTATIONS_FILE_PREFIX,
                                                                                                                             model_idx=initial_model_idx + 1)))

    splitter.split()

    # create a file for full model (filters annottations)
    splitter = DataSplitter(annotation_file_path=os.path.join(ANNOTATIONS_DIR,ORIGINAL_ANNOTATIONS_DIR, ORIGINAL_TRAIN_ANNOTATION_FILE),
                            train_split_percentage=0.99,
                            train_output_file=os.path.join(ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR, ANNOTATIONS_FILE_FULL_MODEL),
                            test_output_file=os.path.join(ANNOTATIONS_DIR, NEW_ANNOTATIONS_DIR, 'empty_annotations_file'))

    splitter.split()

if __name__ == '__main__':
    main()
