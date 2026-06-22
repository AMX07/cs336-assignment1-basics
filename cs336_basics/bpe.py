import os
from typing import BinaryIO
import multiprocessing 
from collections import Counter
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END) # (offset and from where) moves the cursor to the end.
    file_size = file.tell() # cursor is at the end, tell() returns where the cursor is at (int) 
    file.seek(0) # move the cursor back to the end 

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)] # chunk_size * (0...desired_num_chunks + 1)
    chunk_boundaries[-1] = file_size # last boundary at the end of the file.

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk
                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break
                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokn(job:tuple):
        """
        returns the count of pretokens
        """
        input_path, start, end, special_token = job
        counter = Counter()
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            special_token_str = special_token.decode("utf-8")
            docs = re.split(re.escape(special_token_str), chunk)
            for doc in docs:
               for match in re.finditer(PAT,doc):
                   pretoken = match.group()
                   counter[pretoken.encode("utf-8")] += 1              
        # print(len(counter))
        return counter


def bpe_tokn(input_path:str, vocab_size:int, special_token:list[str]):
    
    '''
    computes the merges, and returns the vocab
    '''

    merged_counter = Counter() # contains the counts for all pre-token accross docs or chunks
    char_count = {}
    
    ## chunking for parallel processing.
    with open(input_path, "rb") as f:
        with multiprocessing.Pool(processes=4) as pool:
            num_processes = 4
            boundaries = find_chunk_boundaries(f, num_processes, special_token)
            jobs = [(input_path,start, end, special_token) for start, end in zip(boundaries[:-1], boundaries[1:])]
            counters = pool.map(pre_tokn, jobs)
            for counter in counters:
                merged_counter.update(counter) #pre-token frequency : [cally] = 3

    for k,v in merged_counter.items(): 
        k = tuple(bytes([b]) for b in k)
        char_count[k] = v # 'c', 'l', 'l','y' : 3
    
    #initialize vocab with 255 bytes, should be a dict
    vocab = [bytes([i]) for i in range(256)]
    vocab.append(special_token)

    merges = []

    #Count adjacent pairs from char_count, weighted by pre-token frequency.
    pair_count = Counter()

    for word,count in char_count.items(): 
        for i in range(len(word)-1):
            pair_count[(word[i],word[i+1])] += char_count[word]
    
    # Merge the highest-priority pair.
    best_pair = pair_count.most_common(1)[0][0]
    # Add merged byte sequence to vocab.
    vocab.append(best_pair)
    # Append the pair to merges.
    merges.append(best_pair)
    
    # char_count.key :{..... (b'C', b'a', b'l', b'l', b'y'): 1 .....}


    # Update char_count by replacing that pair wherever it appears. 
    for word in char_count.keys(): 
        for i in range(len(word)-1):
            if word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                print("print(best_pair, word)")
                print(best_pair, word)
                bp = bytes(best_pair[0] + best_pair[1]) # <class 'bytes'> b' t'
                # print(word) tuple to bytes
                # word =  bytes(bp) + word[i+2:] 
                new_word = word[:i] + (bp,) + word[i+2:]
                word = new_word
                print(word,bp,"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(char_count.get(word))
                
            else:
                break
    




                


    
    
    # for k,v in char_count.items():
    #     # if best_pair in k:
    #     print(k,v)

    # Repeat until len(vocab) == vocab_size, or no pairs remain.
    
    # {(b' ', b'c', b'a', b'm', b'e', b'l'): 1, (b'C', b'a', b'l', b'l', b'y'): 1}
    
    #time to write an algo that computes the merges
    '''
    #vocab, merges # dict[int,byte] and list[tuple[bytes,bytes]]
    vocab = {}
    merges = [()]
    for i,i+1 in char_count{}

    wait we already have pre-token counts.
    '''


    
                    
    return None            

    
# pre-tokenization
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


if __name__ == "__main__":
    input_path = "/Users/anshmittal/projects/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    special_token = b"<|endoftext|>"
    bpe_tokn(input_path,256,special_token)