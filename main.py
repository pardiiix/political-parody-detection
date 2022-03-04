import pandas as pd
import torch



def create_pandas_df(path, text_file):
  #creating pandas df a text set
  df = pd.read_csv(f'{path}{text_file}', sep=",",header=None)
  print(f"original txt file {text_file} has a len of", len(df))
  df.columns = ['tweet_id', 'label']

  #add raw tweet to the dataframe (left-join)
  df = df.merge(complete_df, on = 'tweet_id')
  clean_df = df.drop(['tweet_id', 'label_y'], axis = 1)
  clean_df.rename({'label_x': 'label'}, axis=1, inplace=True)
  df = clean_df
  return df


def combine_all_data():
	#need to combine all three csv files together
	split_1 = pd.read_csv(f'{root_path}data_split1.csv', lineterminator='\n')
	split_2 = pd.read_csv(f'{root_path}data_split2.csv', lineterminator='\n')
	split_3 = pd.read_csv(f'{root_path}data_split3.csv', lineterminator='\n')

	complete_df = pd.concat([split_1, split_2, split_3], axis=0)
	print("the complete dataframe:")
	print(complete_df.head())



# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



root_path = './'
gender_path = f'{root_path}gender/'
#training on male data, testing on female data
test_on_f_path = f'{gender_path}test_on_female/'
#testing on male path
test_on_m_path = f'{gender_path}test_on_male/'

combine_all_data()

#creating pandas df of male train and dev set, and female test set
male_train_df = create_pandas_df(test_on_f_path, "male_train.txt")
male_dev_df = create_pandas_df(test_on_f_path, "male_dev.txt")
female_test_df = create_pandas_df(test_on_f_path, "female.txt")
print("a sample of the male training set:")
print(male_train_df.sample(5))
print("len(male_train_df)",len(male_train_df))
print("len(male_dev_df)",len(male_dev_df))
print("len(female_test_df)",len(female_test_df))



# Get the lists of sentences and their labels.
sentences = male_train_df.tweet_pp.values
labels = male_train_df.label.values

val_sentences = male_dev_df.tweet_pp.values
val_labels = male_dev_df.label.values


#######################################################################
#			Tokenization & Input Formatting
#######################################################################

#BERT Tokenizer
from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)