# from multiprocessing import Pool
import os
from typing import BinaryIO


import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(PAT, "some text that i'll pre-tokenize <|endoftext|>")

split_special_token = b"<|endoftext|>"
docs = []

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
    file.seek(0, os.SEEK_END) # moves the reference pointer to the end
    file_size = file.tell( ) # return an integer representing the current position
    file.seek(0) # # moves the reference pointer to the begining

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

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


## Usage
with open('/Users/anshmittal/projects/assignment1-basics/data/owt_valid.txt', "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, split_special_token)
    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    
    chunks =  [(boundaries[:-1], boundaries[1:])]
        # f.seek(start)
        # chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # # Run pre-tokenization on your chunk and store the counts for each pre-toke
        # for i in range(chunk.count("<|endoftext|>") //2):
        #         doc_start = chunk.find("<|endoftext|>")
        #         doc_end = chunk.find("<|endoftext|>", doc_start+1)
        #         doc = chunk[doc_start + 12: doc_end]
        #         # pre-tokenize
       
        # special_tokens = ["<|endoftext|>"]
        # pattern = "|".join(re.escape(tok) for tok in special_tokens)
        # segments = re.split(pattern, chunk)
        
        # for segment in segments:
        #     pre_tokens = [match.group(0) for match in re.finditer(PAT, segment)]
        #     docs.append(pre_tokens)
    print(chunks)
        
