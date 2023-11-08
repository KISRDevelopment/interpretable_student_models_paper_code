import pandas as pd
import numpy as np 
import split_dataset
import json 
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re 
import ast 
from _ast import AST

def main():
    output_name = 'equations'

    df = pd.read_csv("data/raw/ds165_equations.txt", sep="\t")
    skill_col = 'KC (Default)'
    is_hint = df['First Attempt'] == 'hint'
    is_na = pd.isna(df[skill_col])
    is_control = df['Condition'] == 'control'

    df = df[~is_hint & ~is_na & is_control]
    
    problems = extract_problems(df['Step Name'].str.lower().tolist())

    print("Trials: %d" % df.shape[0])
    print("Students: %d" % pd.unique(df['Anon Student Id']).shape[0])
    print("Steps: %d" % pd.unique(problems).shape[0])
    print("Skills: %d" % (pd.unique(df[skill_col]).shape[0]))
    
    problem_id, problem_text_to_id = to_numeric_sequence(problems)
    with open("data/datasets/equations.problem_text_to_id.json", "w") as f:
        json.dump(problem_text_to_id, f, indent=4)
    
    output_df = pd.DataFrame({
        "student" : to_numeric_sequence(df['Anon Student Id'])[0],
        "problem" : problem_id,
        "skill" : to_numeric_sequence(df[skill_col])[0],
        "correct" : [1 if c == 'correct' else 0 for c in df['First Attempt']]
    })

    output_df.to_csv("data/datasets/%s.csv" % output_name, index=False)
    full_splits = split_dataset.main(output_df)
    np.save("data/splits/%s.npy" % output_name, full_splits)
    
    problem_id_to_text = { v: k for k, v in problem_text_to_id.items() }
    problems = [problem_id_to_text[i] for i in range(len(problem_id_to_text))]
    
    model = SentenceTransformer('all-mpnet-base-v2')
    embedding = model.encode(problems)
    print(embedding.shape)
    np.save("data/datasets/%s.embeddings.npy" % output_name, embedding)

def extract_problems(steps):
    new_steps = []
    for s in steps:
        parts = re.split(r'\s*=\s*', s)
        if len(parts) == 2:
            left = parse_expr(parts[0])
            right = parse_expr(parts[1])
            parsed = "%s = %s" % (left, right)
            new_steps.append(parsed)
        else:
            new_steps.append(s)
    
    return new_steps

def parse_expr(expr):

    # norm var names
    expr = expr.replace('y', 'x')

    # handle implict multiplications
    expr = re.sub(r'(\d)x', r'\1*x', expr)

    # parse into python ast
    parsed = ast.parse(expr)
    
    # replace constants
    transformer = Transformer()
    transformer.visit(parsed)
    
    return ast.unparse(parsed)

def to_numeric_sequence(vals):

    sorted_vals = sorted(set(vals))

    mapping =  dict(zip(sorted_vals, range(len(vals))))

    return [mapping[v] for v in vals], mapping


class Transformer(ast.NodeTransformer):

    def generic_visit(self, node: AST) -> AST:
        if node.__class__.__name__ == "Constant":
            return ast.Name('C')
    
        return super().generic_visit(node)

if __name__ == "__main__":
    main()