"""
Running this preprocessing.py as main will do the following:

1. Reads all the plays under data/shakespeares-works_TXT_FolgerShakespeare
2. Strip all the comments before "Characters in the Play" (if any)
3. Remove all the lines of "=" separator in each play
4. Concatenate all the players into one single txt file

Note: The headers of plays without "Characters in Play" are removed manually.
"""
import logging

from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preprocess")


def read_play_file(file_path):
  with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()
  return text


def remove_headers(text):
  """Remove headers before the character introduction 'Characters in the Play'"""
  start_ind = text.find("Characters in the Play")
  if start_ind >= 0:
    return text[start_ind:]
  else:
    # headers already removed manually.
    return text


def remove_separator_lines(s):
  """Remove separator line of '='. """
  return '\n'.join(
    line for line in s.splitlines()
    if not line or set(line) != {'='}
  )


def store_corpus(corpus, output_path):
  with open(output_path, "w", encoding="utf-8") as f:
    f.write(corpus)


def main():
  data_path = Path("data/shakespeares-works_TXT_FolgerShakespeare")
  output_path = Path("data/output.txt")
  output = ""
  for file_path in data_path.glob("*.txt"):
    logger.info(f"Begin reading {file_path}")
    text = read_play_file(file_path)
    text = remove_headers(text)
    text = remove_separator_lines(text)
    output = output + '\n' + text
    logger.info("Done appending.")
  output = output[1:]  # remove the first \n
  store_corpus(output, output_path)


if __name__ == "__main__":
  logger.info("Begin preprocessing plays.")
  main()
  logger.info("Preprocessing done.")