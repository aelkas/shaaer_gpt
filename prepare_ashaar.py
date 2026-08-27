"""
Build a POEM-STRUCTURED Arabic corpus from ARBML/Ashaar.

Why: MetRec is verse-level and shuffled -- consecutive lines come from
different poems by different poets, so a model trained on it CANNOT learn
anything that spans a poem.  Ashaar keeps poems intact.

Schema of arbml/ashaar (254,630 poems, 3.86M baits, 7,167 poets):
    poem title    str
    poem meter    str    e.g. "بحر الخفيف"   (101 classes)
    poem verses   list[str]  <-- FLAT LIST OF HEMISTICHS, not full verses.
                                 verses[0] is a sadr, verses[1] its ajuz,
                                 verses[2] the next sadr, and so on.
    poem theme    str    e.g. "قصيدة مدح"    (18 classes)
    poet name     str
    poet era      str    e.g. "العصر العثماني" (14 classes)
    poet location str

LICENCE WARNING.  Ashaar is released "under fair use for research development
only ... should not be used for any commercial purposes", and poets may
request takedown.  This is weaker than MetRec's MIT licence, and Ashaar
includes modern (in-copyright) poets.  Use `eras=` to restrict to pre-modern
poets if that matters for your use.

Output format, one poem per block, blank line between poems:

    [متقارب|مدح]
    سدر الأول # عجز الأول
    صدر الثاني # عجز الثاني

    [كامل|غزل]
    ...
"""

import re
import unicodedata

TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
TATWEEL = "\u0640"
ARABIC = "\u0621-\u063A\u0641-\u064A\u0671"
ALLOWED = re.compile(rf"[^{ARABIC}\u064B-\u0652\u0670 ]")
ORPHAN = re.compile(r"(?:(?<=^)|(?<=\s))[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]+")

# pre-modern eras, for a roughly public-domain subset
CLASSICAL_ERAS = {
    "العصر الجاهلي", "المخضرمين", "العصر الاسلامي", "العصر الاموي",
    "العصر العباسي", "العصر الايوبي", "العصر المملوكي", "العصر العثماني",
    "العصر الفاطمي", "العصر الاندلسي",
}


def clean(s, keep_diacritics=False):
  s = unicodedata.normalize("NFKC", s).replace(TATWEEL, "")
  if keep_diacritics:
    s = ORPHAN.sub("", s)
  else:
    s = TASHKEEL.sub("", s)
  s = re.sub("[إأآٱ]", "ا", s)
  s = s.replace("ى", "ي").replace("ة", "ه")
  s = ALLOWED.sub(" ", s)
  return re.sub(r"\s+", " ", s).strip()


def short_meter(m):
  """'بحر مجزوء الرمل' -> 'مجزوء الرمل'"""
  return (m or "").replace("بحر ", "").strip() or "غير معروف"


def short_theme(t):
  """'قصيدة مدح' -> 'مدح'"""
  return (t or "").replace("قصيدة ", "").strip() or "عامه"


def build_corpus(ds, keep_diacritics=False, conditioning=True,
                 eras=None, min_baits=3, max_baits=60, limit=None):
  """
  ds: an iterable of dicts with the Ashaar schema.
  eras: set of allowed `poet era` values, or None for all.
  min_baits/max_baits: drop fragments and enormous outliers.
  conditioning: prepend a [meter|theme] header line to each poem, so the
      model can learn to condition on them -- and so you can prompt with one.
  Returns (text, stats).
  """
  blocks, stats = [], dict(seen=0, kept=0, no_meter=0, odd=0, too_short=0,
                           too_long=0, wrong_era=0, baits=0)

  for rec in ds:
    stats["seen"] += 1
    if limit and stats["kept"] >= limit:
      break

    if eras is not None and rec.get("poet era") not in eras:
      stats["wrong_era"] += 1
      continue

    hemis = rec.get("poem verses") or []
    if len(hemis) % 2:            # truncated scrape: drop the dangling half
      stats["odd"] += 1
      hemis = hemis[:-1]

    baits = []
    for i in range(0, len(hemis), 2):
      a, b = clean(hemis[i], keep_diacritics), clean(hemis[i+1], keep_diacritics)
      if len(a.split()) < 2 or len(b.split()) < 2:
        continue
      baits.append(f"{a} # {b}")

    if len(baits) < min_baits:
      stats["too_short"] += 1
      continue
    if len(baits) > max_baits:
      baits = baits[:max_baits]
      stats["too_long"] += 1

    lines = []
    if conditioning:
      lines.append(f"[{short_meter(rec.get('poem meter'))}|"
                   f"{short_theme(rec.get('poem theme'))}]")
    lines += baits
    blocks.append("\n".join(lines))
    stats["kept"] += 1
    stats["baits"] += len(baits)

  text = "\n\n".join(blocks) + "\n"
  stats["chars"] = len(text)
  stats["vocab"] = len(set(text))
  return text, stats
