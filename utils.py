import tensorflow as tf
def min_seq_len_filter(df, val):
    
    by_student = df.groupby('student').size()
    by_student = by_student[by_student >= val]
    new_df = df[df['student'].isin(set(by_student.index))]

    return new_df

def xe_loss(ytrue, ypred, mask):
    """
        ytrue: [n_batch, n_steps]
        ypred: [n_batch, n_steps]
        mask:  [n_batch, n_steps]
    """
    losses = -(ytrue * tf.math.log(ypred) + (1-ytrue) * tf.math.log(1-ypred))
    losses = losses * mask
    return tf.reduce_sum(losses) / tf.reduce_sum(mask)

def load_params(components, params):
    for component in components:
        for name, p in component.trainables:
            p.assign(params[name])
   
def save_params(components):
    params = {}
    for component in components:
        for name, p in component.trainables:
            params[name] = p.numpy()
    return params