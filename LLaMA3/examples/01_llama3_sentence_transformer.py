from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

sentence = ['This framework generates embeddings for each input sentence']

embedding = model.encode(sentence)