#learning llms from scratch
#1. BPE Tokenization (Imperative Approach)

print("*" * 45)

text = "low lower lowest café"
print(f"original text: {text}")

# Encode string into UTF-8 bytes (variable length: 1-4 bytes per character)
utf8_bytes = text.encode("utf-8")
print(f"UTF-8 bytes: {utf8_bytes}")

# Convert bytes object into a list of integer byte IDs (0-255)
byte_ids = list(utf8_bytes)
print(f"Initial byte-level token IDs:")
print(byte_ids)

print(f"Character count: {len(text)} | Byte ID count: {len(byte_ids)}")

print("*" * 45)

#finding pair counts
pair_counts = {}
for i in range(len(byte_ids)-1):
    pair = (byte_ids[i], byte_ids[i+1])
    #print(pair)
    if pair in pair_counts:
        pair_counts[pair] += 1
    else:
        pair_counts[pair] = 1
print(pair_counts)


#finding best pair
best_pair = max(pair_counts, key=pair_counts.get)
print(best_pair)
print(bytes([best_pair[0]]).decode("utf-8")+bytes([best_pair[1]]).decode("utf-8"))

#merging best pair
new_id = 256
merged_ids = []
i = 0

while i < len(byte_ids):
    if i < len(byte_ids) - 1 and ((byte_ids[i], byte_ids[i+1]) == best_pair):
        merged_ids.append(new_id)
        i += 2
    else:
        merged_ids.append(byte_ids[i])
        i += 1

print(merged_ids)
print(f"length of byte ids: {len(byte_ids)}")
print(f"length of merged ids: {len(merged_ids)}")

#all above steps in a loop for num of merges
# repeate this whole process for n number of merges

text = input("provide train text: ")
#step 1: byte encoding of the text
ids = text.encode("utf-8")
print(ids)
ids = list(ids)
print(ids)

#BPE merges loop
merges = {}
num_merges = int(input("provide num of merges: "))

for step in range(num_merges):
    #step 2: find pair counts
    pair_counts = {}
    for i in range(len(ids)-1):
        pair = (ids[i], ids[i+1])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    if not pair_counts:
        print(f"No pairs left to merge at step {step}; stopping early.")
        break
        
    #step 3: find best pair
    best_pair = max(pair_counts, key=pair_counts.get)

    if pair_counts[best_pair] < 2:
        print(f"No pair occurs more than once at step {step}; stopping early.")
        break

    #step 4: do the merges
    new_id = 256 + step
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ((ids[i], ids[i+1]) == best_pair):
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
            
    ids = new_ids
    merges[best_pair] = new_id
    print(f"step {step}: merge {best_pair} -> {new_id}, count: {pair_counts[best_pair]}, final length of ids now: {len(ids)}")
    print(ids)

print(f"merges: {merges}")
print(f"final ids: {ids}")