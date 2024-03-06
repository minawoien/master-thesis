import numpy as np

def choose_plaintext(batch_size, p1_bits, p2_bits):
    p1a_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
    p1b_batch = np.random.randint(0, 2, p1_bits * batch_size).reshape(batch_size, p1_bits)
    p2a_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)
    p2b_batch = np.random.randint(0, 2, p2_bits * batch_size).reshape(batch_size, p2_bits)

    p1_batch = np.zeros_like(p1a_batch)
    p2_batch = np.zeros_like(p2a_batch)

    choices = np.zeros((batch_size, 4), dtype=int)

    for i in range(batch_size):
        p1_choice = np.random.rand() < 0.5
        p2_choice = np.random.rand() < 0.5

        p1_batch[i] = p1a_batch[i] if p1_choice else p1b_batch[i]
        p2_batch[i] = p2a_batch[i] if p2_choice else p2b_batch[i]

        choices[i, 0] = 1 if p1_choice else 0  
        choices[i, 1] = 0 if p1_choice else 1  
        choices[i, 2] = 1 if p2_choice else 0
        choices[i, 3] = 0 if p2_choice else 1
    
    return {"p1": p1_batch, "p2": p2_batch, "p1a": p1a_batch, "p1b": p1b_batch, "p2a": p2a_batch, "p2b": p2b_batch, "choices": choices}

if __name__ == "__main__":
    batch_size = 5  # Example batch size
    p1_bits = 16  # Example bit size for p1
    p2_bits = 16  # Example bit size for p2
    p = choose_plaintext(batch_size, p1_bits, p2_bits)
