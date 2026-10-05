from typing import Optional
import re
import unicodedata

__all__ = ["TextProcess"]

def half_full_translate(text: str, mode: Optional[str]=None) -> str:
    """英数字の半角全角を変換する。

    Args:
        text (str): 変換したい文字列。
        mode (Optional[str], optional):
            None: なにもしない。
            'half': 半角に変換する。
            'full': 全角に変換する。

    Returns:
        str: 変換後の文字列。
    """
    if mode is None:
        return text
    elif mode == "half":
        result = ""
        for char_ord in map(ord, text):
            if ord("ａ") <= char_ord <= ord("ｚ"):
                result += chr(char_ord - ord("ａ") + ord("a"))
            elif ord("Ａ") <= char_ord <= ord("Ｚ"):
                result += chr(char_ord - ord("Ａ") + ord("A"))
            elif ord("０") <= char_ord <= ord("９"):
                result += chr(char_ord - ord("０") + ord("0"))
            else:
                result += chr(char_ord)
        return result
    elif mode == "full":
        result = ""
        for char_ord in map(ord, text):
            if ord("a") <= char_ord <= ord("z"):
                result += chr(char_ord - ord("a") + ord("ａ"))
            elif ord("A") <= char_ord <= ord("Z"):
                result += chr(char_ord - ord("A") + ord("Ａ"))
            elif ord("0") <= char_ord <= ord("9"):
                result += chr(char_ord - ord("0") + ord("０"))
            else:
                result += chr(char_ord)
        return result
    else:
        raise ValueError(f"{mode} is unknown.")

def remove_marks(text: str) -> str:
    return re.sub(
        "[\\-,.。、？！：；＜＞（）｛｝［］〈〉《》「」『』【】“”‘’▲△▼▽■□◆◇●○…‥・　＿\\?\\!:;<>\\(\\)\\{\\}\"'_]",
        " ",
        text
    )

class TextProcess:
    def __init__(
        self, normalize: bool=False, case: Optional[str]=None,
        half_full: Optional[str]=None, remove_marks: bool=False
    ):
        """
        漢数字の変換は行わない。

        Args:
            normalize (bool, optional):
                NFKCによるunicode正規化を行う。
                カタカナは全角、英数字は半角になる。特殊な記号も分解される。
            case (Optional[str], optional):
                アルファベットの大文字小文字を変換する。'lower', 'upper' または None
            half_full (Optional[str], optional):
                英数字の全角半角を変換する。この変換はnormalizeより優先される。
                'half', 'full' または None
            remove_marks (bool, optional):
                記号を削除する。句読点も含む。

        Examples:
            >>> normalizer = TextProcess(normalize=True)
            >>> normalizer("100 １００ abc ａｂｃ ㌖ ㎞ ① № ㈱")
                '100 100 abc abc キロメートル km 1 No (株)'
            >>> normalizer = TextProcess(normalize=True, case='upper', half_full='full', remove_marks=True)
            >>> normalizer("100 １００ abc ａｂｃ ㌖ ㎞ ① № ㈱")
                '１００ １００ ＡＢＣ ＡＢＣ キロメートル ＫＭ １ ＮＯ 株'
        """
        self.tag_ptrn     = re.compile("(?<= )(%[^%]+%|％[^％]+％)(?= )")
        self.normalize    = normalize
        self.case         = case
        self.half_full    = half_full
        self.remove_marks = remove_marks


    def __call__(self, text: str) -> str:
        text = f" {text} " # 処理の都合上先頭と末尾に半角スペースを入れておく。

        # タグを削除。
        text = self.tag_ptrn.sub("", text)

        # カタカナは全角に変換。英数字は半角に変換。
        # ㌖とか①とかも分解する。
        if self.normalize:
            text = unicodedata.normalize("NFKC", text)

        # 英数字をself.half_fullにしたがって変換する。
        text = half_full_translate(text, self.half_full)

        # 記号を削除する。句読点も消える。
        if self.remove_marks:
            text = remove_marks(text)

        # 大文字または小文字にそろえる。
        if self.case == "lower":
            text = text.lower()
        elif self.case == "upper":
            text = text.upper()

        # 不要な空白文字を削除する。
        text = re.sub("^\\s+|\\s+$", "", text)
        text = re.sub("\\s+", " ", text)

        return text
