# task8
Constructing HF transformer dataset with inline mp3-to-wav converter


# Remarks @ 2022-03-11T12:24:14

I acquired a brief .mp3 file and tried converting it with ffmpeg: `ffmpeg -i sample.mp3 -ar 16000 sample.wav`. It turns out the difference in filesizes is not as dramatic as internet had me believe: instead of 10x difference we only see 2.6x reduction in filesize.

When implementing the same functionality in python with `pydub`, the file size is precisely the same as before.

Next step will be dataset preparation with inline conversion.


# Addendum 2022-03-11T14:20:44

The solution has been proposed in [the notebook](1.ipynb), but so far has not been tested:

```python
import tempfile
from pydub import AudioSegment
def load_audio(path):
    with tempfile.NamedTemporaryFile() as f:
        AudioSegment.from_mp3(path).export(
                f.name,
                format="wav", 
                parameters=["-ar", "16000"]
                )
        return datasets.Audio(sampling_rate=16000).decode_example(f.name)
```

## Conversion speed

Based on my one-shot estimation for a 8.7s long sample I estimate we need about 19 hours for the conversion alone. 

# Moving on

As per Nikola's email I start working on the generation of a random sample of wavs. 

Meeting notes:
* Use reconames as well (also for audio files).
* Open a google sheet (so export as excel and rsync to laptop.)

Ok, due to weird glitches in the file I/O about 30% of the files won't open. For now I just pass over them. I can process more later and get the sample count to 1k.

