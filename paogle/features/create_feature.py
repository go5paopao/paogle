import time
import logging
from pathlib import Path
from abc import ABCMeta, abstractmethod
import numpy as np
from paogle.utils.file_io import save_pickle, load_pickle
from paogle.utils.dataframe import fast_concat


class CreateFeature(metaclass=ABCMeta):
    def __init__(self, reset=False, merge=True):
        self.save_path = Path("../feature/")
        self.reset = reset
        self.merge = merge

    def _check_before_cols(self, train_df, test_df):
        self.train_before_cols = train_df.columns.tolist()
        self.test_before_cols = test_df.columns.tolist()

    def _check_after_cols(self, train_df, test_df):
        self.train_after_cols = train_df.columns.tolist()
        self.test_after_cols = test_df.columns.tolist()

    def _get_new_cols(self, train_df, test_df):
        train_new_cols = np.setdiff1d(self.train_after_cols,
                                      self.train_before_cols)
        test_new_cols = np.setdiff1d(self.test_after_cols,
                                     self.test_before_cols)
        assert(len(np.setdiff1d(train_new_cols, test_new_cols)) == 0)
        return train_new_cols

    def _save_feature(self, train_df, test_df):
        new_cols = self._get_new_cols(train_df, test_df)
        self.train_path.parent.mkdir(exist_ok=True)
        self.test_path.parent.mkdir(exist_ok=True)
        if self.merge:
            save_pickle(train_df[new_cols], self.train_path)
            save_pickle(test_df[new_cols], self.test_path)
        else:
            save_pickle(train_df, self.train_path)
            save_pickle(test_df, self.test_path)

    def _exists_feature(self):
        class_name = self.__class__.__name__
        self.train_path = self.save_path / \
            "train" / "{}.pkl".format(class_name)
        self.test_path = self.save_path / "test" / "{}.pkl".format(class_name)
        if self.train_path.is_file() and self.test_path.is_file():
            return True
        else:
            return False

    def load_and_merge(self, train_df, test_df, start):
        train_feat = load_pickle(self.train_path)
        test_feat = load_pickle(self.test_path)
        if self.merge:
            train_df = fast_concat(train_df, train_feat)
            test_df = fast_concat(test_df, test_feat)
        else:
            train_df = train_feat
            test_df = test_feat
        logging.info("load complete ... {:.1f}s".format(time.time()-start))
        return train_df, test_df

    def make(self, train_df, test_df, *args):
        # check whether feature has saved or not
        logging.info("******** {} ********".format(self.__class__.__name__))
        if self._exists_feature() and not self.reset:
            logging.info("loading...")
            start = time.time()
            return self.load_and_merge(train_df, test_df, start)
        else:
            logging.info("making...")
            start = time.time()
            # save before cols
            self._check_before_cols(train_df, test_df)
            # feature make
            train_df, test_df = self.__call__(train_df, test_df, *args)
            # save after cols
            self._check_after_cols(train_df, test_df)
            # save feature
            self._save_feature(train_df, test_df)
            logging.info("make complete ... {:.1f}s".format(time.time()-start))
            return train_df, test_df

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()
