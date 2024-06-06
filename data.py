import os
import json
from bisect import bisect
from typing import Union
from abc import ABC, abstractmethod
from functools import cache
from collections import Counter
import torch

GRAPH_PATH = os.path.join(os.path.dirname(__file__),"graph.json")

class LabelBase(ABC):
    def __init__(self, name: str, id=None):
        self.name = name
        self.id = id

    @abstractmethod
    @cache
    def get_dirs(self) -> list[str]:
        pass
    
    def __getitem__(self, index):
        return self.get_dirs()[index]

    def __len__(self):
        return len(self.get_dirs())

    def __str__(self):
        return self.name

class Label(LabelBase):
    def __init__(self, name: str, id=None, parent=None, childs: list = None):
        super().__init__(name, id)
        if childs is None:
            self.childs = []
        else:
            self.childs = childs
        self.parent = parent

    def __dict__(self):
        return {
            "type": "Label",
            "data": {
                "id": self.id,
                "name": self.name,
                "parent": self.parent,
                "childs": self.childs,
            },
        }

    def get_dirs(self) -> list[str]:
        if self.parent and not self.childs:
            return [self.parent + "/" + self.name]
        elif self.childs:
            return [self.name + "/" + c for c in self.childs]

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.parent == other.parent
            and self.childs == other.childs
        )

    def __repr__(self):
        return f"Label(name={self.name}, id={self.id})"

class CustomLabel(LabelBase):
    def __init__(self, name, labels: dict[str, list[str]], id: int = None):
        
        super().__init__(name, id)
        # 按照键排序
        sorted_keys = sorted(labels.keys())
        self.labels = {key: sorted(labels[key]) for key in sorted_keys}

    def __dict__(self):
        return {
            "type": "CustomLabel",
            "data": {"id": self.id, "name": self.name, "labels": self.labels},
        }

    def get_dirs(self) -> list[str]:
        ret = []
        for p, c in self.labels.items():
            ret.extend([p + "/" + c_ for c_ in c])
        return ret

    def __repr__(self):
        return f"CustomLabel(name={self.name}, id={self.id})"
    
    def get_graph(self):
        graph_str = ""
        graph_str += f"{self.name}\n"
        for p, c in self.labels.items():
            graph_str += f"  |\n"
            graph_str += f"  |->{p}\n"
            graph_str += f"     |\n"
            for cc in c:
                graph_str += f"     |->{cc}\n"
        return graph_str

class Category:
    def __init__(self, labels: list[LabelBase], name: str = None):
        self.name = name
        if labels is None:
            self.labels = []
        else:
            self.labels = labels

        self.__sort__()

    def __sort__(self):
        # 按name排序labels
        self.labels.sort(key=lambda x: x.name)

        self.bisert_breakpoints = []
        start = 0
        # 设置labels的id
        for i, label in enumerate(self.labels):
            label.id = i
            start += len(label)
            self.bisert_breakpoints.append(start)

    def get_label_index(self, index: int):
        return bisect(self.bisert_breakpoints, index)

    def append(self, label: LabelBase):
        self.labels.append(label)
        self.__sort__()
       
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]

    def __dict__(self):
        return {
            "name": self.name,
            "labels": [label.__dict__() for label in self.labels],
        }

    def load_graph(graph_path: str = GRAPH_PATH):
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
        return graph

    def dump_json(self, save_path: str):
        with open(save_path, "w") as f:
            json.dump(self.__dict__(), f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, "r") as f:
            js = json.load(f)
        name = js["name"]
        labels = js['labels']
        labels = [globals()[l['type']](**l['data']) for l in labels]
        return cls(name=name, labels=labels)

    def check_valid(self):
        dirs = {}
        all_dirs = []
        for label in self.labels:
            dirs[label.name] = label.get_dirs()
            all_dirs.extend(dirs[label.name])
        # 检查all_dirs是否有重复的目录
        if len(set(all_dirs)) != len(all_dirs):
            counter = Counter(all_dirs)
            duplicates = [num for num, count in counter.items() if count > 1]
            return False, duplicates
        return True, None
        
    @classmethod
    def from_selections(
        cls,
        selections: Union[str, list[str]],
        custom_labels: Union[CustomLabel,list[CustomLabel]]=None,
        name: str = None,
    ):
        if isinstance(selections, str):
            selections = [selections]
        assert len(selections) > 0, "最少选择一个分类"

        graph = cls.load_graph()

        if custom_labels is None:
            custom_labels = []
            
        if isinstance(custom_labels, CustomLabel):
            custom_labels = [custom_labels]
            
        labels = [c for c in custom_labels]
        
        # 从graph中移除掉custom_labels中的分类
        for custom_label in custom_labels:
            for cp, cc in custom_label.labels.items():
                new_list = [c for c in graph[cp] if c not in cc]
                graph[cp] = new_list
        
        childs = [s for s in selections if s not in graph.keys()]
        parents = [s for s in selections if s in graph.keys()]

        if len(childs) == 0:
            assert len(parents) > 0, "最少选一个分类"  # 1. c == 0, p==0
            if len(parents) == 1:
                labels.extend([
                    Label(name=child, parent=parents[0]) for child in graph[parents[0]]
                ])  # 2.c==0, p==1 -> 只有一个父类，添加子类
            elif len(parents) > 1:
                labels.extend([
                    Label(name=p, childs=graph[p]) for p in parents
                ])  # 3.c==0, p>1 -> 多个父类，添加父类

        elif len(childs) > 0:
            if len(childs) == 1:
                assert (
                    len(parents) > 0
                ), "只有一个分类，至少选择两个分类"  # 4. c==1, p==0

            for c in childs:  # 5. c>=1, p>=0 -> 添加多个子类， 多个父类
                for parent, child in graph.items():
                    if c in child:
                        labels.append(Label(name=c, parent=parent))  # 添加子分类
                        graph[parent].remove(
                            c
                        )  # 父分类的子分类列表中删除子分类, 所以要用拷贝的graph
            for p in parents:
                labels.append(Label(name=p, childs=graph[p]))  # 添加父分类

        ret = cls(name=name, labels=labels)
        return ret

    def __repr__(self):
        return f'Category(name="{self.name}", labels={self.labels})'

class MyDataset(torch.utils.data.Dataset):
# class MyDataset():
    # 1200 files in each dir
    # dataset -> category -> labels -> dirs -> files
    def __init__(self, category: Category, base_dir: str):
        self.category = category
        self.base_dir = base_dir

    def __len__(self):
        return sum(len(label) for label in self.category.labels) * 1200

    def __getitem__(self, index):
        if index < 0:
            index = len(self) + index

        file_idx = index % 1200
        dir_idx = index // 1200
        label_idx = self.category.get_label_index(dir_idx)
        
        dir_idx = dir_idx - self.category.bisert_breakpoints[label_idx - 1] if label_idx > 0 else dir_idx

        """
        label_index
        dir_index
        file_index
        """
        label = self.category[label_idx]
        dir_ = label[dir_idx]
        file_ = os.path.join(self.base_dir, dir_, f"{label.name}_{file_idx}.wav")
        # signal, sr = torchaudio.load(file_)
        return file_, label.id


def main():
    s = ["交通噪声", "狗叫声", "社会噪声"]
    c = Category.from_selections(s, name="环境噪声")
    i = 21

    d = MyDataset(c, "data")
    l = len(c)

    # a = d[100]
    # b = d[1200]
    e = d[i * 1200 - 1]
    # e = d[l-1]
    print("ok")


if __name__ == "__main__":
    main()
