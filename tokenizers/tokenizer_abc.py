from abc import ABC, abstractmethod


class TokenizerABC(ABC):
  @abstractmethod
  def train(self, corpus, **kwargs):
    pass

  @abstractmethod
  def save(self, path):
    pass

  @classmethod
  @abstractmethod
  def load(cls, path):
    pass

  @abstractmethod
  def encode(self, corpus):
    pass

  @abstractmethod
  def decode(self, token_ids):
    pass

  @property
  @abstractmethod
  def vocab_size(self):
    pass