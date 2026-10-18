from tokenizers.char_tokenizer import CharTokenizer


def test_chartokenizer():
  corpus = "This is a very long and profoundly philosophical sentence."
  t = CharTokenizer()
  t.train(corpus)

  token_ids = t.encode(corpus)
  decoded_corpus = t.decode(token_ids)
  assert len(corpus) == len(decoded_corpus), "text corpus should have same length"
  for i in range(len(corpus)):
    assert corpus[i] == decoded_corpus[i], "decoded corpus should have the same content"
