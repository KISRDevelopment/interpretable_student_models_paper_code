import openai 
import pandas as pd 
import sys 
import time 
import numpy as np

def main(path, output_path):

    with open('api_key', 'r') as f:
        api_key = f.read()
    
    openai.api_key = api_key

    questions_df = pd.read_excel(path)

    embeddings = []
    i = 0
    for r in questions_df.itertuples():
        text = r.Requirement
        embedding = get_embedding(text)
        embeddings.append(embedding)
        print("Processed %d" % i)
        i += 1
        time.sleep(5)

    embeddings = np.array(embeddings)

    embeddings_df = pd.DataFrame(data=embeddings, columns=np.arange(embeddings.shape[1]).astype(str))
    embeddings_df['AssignmentID'] = questions_df['AssignmentID']
    embeddings_df['ProblemID'] = questions_df['ProblemID']

    embeddings_df.to_csv(output_path, index=False)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.strip().replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
 
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
