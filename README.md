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



# Addendum 2022-03-14T14:17:38

It was noticed that the `FileNotFoundError` was being raised due to the inconsistent whitespace use in the mapping. I managed to remove most of them, for all of them the implementation will follow later.

# Meeting notes 2022-03-14T15:36:25
* Min sim: 0.8
* `from Levenshtein import distance`
* ```def sim(a, b):
    return 1-(distance(a,b)*2/(len(a)+len(b)))```
* Prepare the ALL the data for training: prefered nomenclature: `[youtube hash]_123.4-135.4`
* Keep in mind what instances were sent for sampling.
* How many have sim < 0.8? This could be test set.
* ~~One yt file could be dev, the rest train.~~ Take random subset for dev split. 2k samples.

During the preparation I take the following notes:
* YT hashes are all 11 characters long. The character set has minuses, underscores, ... but not slashes, hashes, dots.
* With my improved error correction I can find all the true filenames. The function has been improved so that I can do the entire corpus in 0.8s instead of 3min45s (24.5dB speedup!)
* 94% of the transcripts satisfy the `sim > 0.8` condition.
* Splitting was performed with a random seed, this way it is more controllable.


At 2022-03-14T17:28:10 the processing was started. It seems to be going really fast. Probably this is due to the optimizations; only one file segment is opened at one time and we have no conversion to mp3. I don't yet know which factor is more important, but would love to know. I estimate the processing will be done in 2.23 hours, which is just as fast as the sample dataset generation.


# Addendum 2022-03-14T18:59:38

So far everything is going as planned. I estimate 39 minutes are left.

# Addendum 2022-03-15T07:32:18

The final size of the corpus is 1.3TB. I opted for generation of a subset in order to be able to fit it all on kt-vm-1tb. I copied the files to a separate folder, which took quite long, and then I started rsyncing data to kt-vm-1TB.

Meeting notes:
* ✓Drop instances where sim < 0.8. Move more data.
* ✓Prepare a new vocabulary.
* ✓Drop what was used in the sample.
* Dev: use 500 samples.
* Train: as much as feasible.
* Test: will be given later.

# Addendum 2022-03-15T12:27:38
Ok, we filled the disk to 97%, I could go higher, but not significantly due to the requirements of saved models and other users.

I started my first training set (directory `6_`). I notice the data loading take quite a long time, probably due to the fact that the dataset is bigger. So far I've not even read the data after 65 min....

# Addendum 2022-03-15T14:44:30

Loading the dataset takes a long time. I experimented with only 1% of the dataset and found out it takes 6 min to process it in a notebook, meaning that I'd need 10 hours to do it. The bottle neck is the part

```python

import datasets
from datasets import load_dataset, load_metric, Audio
def load_audio(path):
    return datasets.Audio(sampling_rate=16000).decode_example(path)

# Adding audio
common_voice_train_df.loc[:, "audio"] = common_voice_train_df.path.apply(load_audio)
```

Running it in a script instead of in a notebook doesn't change the performance significantly.

After a skype chat with Nikola it had been found that the data was sampled with 48kHz frequency instead of 16. That was corrected and new files are being transfered as we speak.

After the files have been transfered,  I started a small trial run to see if it runs OK. It does, it seems that with only a 1k subset the loading and training runs without problems. It takes 20 minutes to run, meaning that we can expect 3 days of training for the full dataset...

To test if everything works as it should I ran another round, this time with 3k instances.

✓ Since the training won't run, go for 200hours of data.

Unnormalized transcripts: /home/korzinek/kaldi/exp/ali_all/unnorm.json
Nikola suggests to do lowercasing and removal of puctiation. Compare with normalized performance. Once the training works of course.

✓ Transform the corpus to mono!

# Addendum 2022-03-18T11:20:56

The training still doesn't run. It seems the limiting factor is not the training, but the dataset loading. I've limited the data to 33k instances. This made it to training, but then failed:

```python
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
~/anaconda3/lib/python3.8/site-packages/wandb/sdk/wandb_init.py in init(job_type, dir, config, project, entity, reinit, tags, group, name, notes, magic, config_exclude_keys, config_include_keys, anonymous, mode, allow_val_change, resume, force, tensorboard, sync_tensorboard, monitor_gym, save_code, id, settings)
    869         try:
--> 870             run = wi.init()
    871             except_exit = wi.settings._except_exit

~/anaconda3/lib/python3.8/site-packages/wandb/sdk/wandb_init.py in init(self)
    441         backend = Backend(settings=s, manager=manager)
--> 442         backend.ensure_launched()
    443         backend.server_connect()

~/anaconda3/lib/python3.8/site-packages/wandb/sdk/backend/backend.py in ensure_launched(self)
    167 
--> 168         self.record_q = self._multiprocessing.Queue()
    169         self.result_q = self._multiprocessing.Queue()

~/anaconda3/lib/python3.8/multiprocessing/context.py in Queue(self, maxsize)
    102         from .queues import Queue
--> 103         return Queue(maxsize, ctx=self.get_context())
    104 

~/anaconda3/lib/python3.8/multiprocessing/queues.py in __init__(self, maxsize, ctx)
     41         self._reader, self._writer = connection.Pipe(duplex=False)
---> 42         self._rlock = ctx.Lock()
     43         self._opid = os.getpid()

~/anaconda3/lib/python3.8/multiprocessing/context.py in Lock(self)
     67         from .synchronize import Lock
---> 68         return Lock(ctx=self.get_context())
     69 

~/anaconda3/lib/python3.8/multiprocessing/synchronize.py in __init__(self, ctx)
    161     def __init__(self, *, ctx):
--> 162         SemLock.__init__(self, SEMAPHORE, 1, 1, ctx=ctx)
    163 

~/anaconda3/lib/python3.8/multiprocessing/synchronize.py in __init__(self, kind, value, maxvalue, ctx)
     79             from .resource_tracker import register
---> 80             register(self._semlock.name, "semaphore")
     81             util.Finalize(self, SemLock._cleanup, (self._semlock.name,),

~/anaconda3/lib/python3.8/multiprocessing/resource_tracker.py in register(self, name, rtype)
    146         '''Register name of resource with resource tracker.'''
--> 147         self._send('REGISTER', name, rtype)
    148 

~/anaconda3/lib/python3.8/multiprocessing/resource_tracker.py in _send(self, cmd, name, rtype)
    153     def _send(self, cmd, name, rtype):
--> 154         self.ensure_running()
    155         msg = '{0}:{1}:{2}\n'.format(cmd, name, rtype).encode('ascii')

~/anaconda3/lib/python3.8/multiprocessing/resource_tracker.py in ensure_running(self)
    120                         signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)
--> 121                     pid = util.spawnv_passfds(exe, args, fds_to_pass)
    122                 finally:

~/anaconda3/lib/python3.8/multiprocessing/util.py in spawnv_passfds(path, args, passfds)
    451     try:
--> 452         return _posixsubprocess.fork_exec(
    453             args, [os.fsencode(path)], True, passfds, None, None,

OSError: [Errno 12] Cannot allocate memory

The above exception was the direct cause of the following exception:

Exception                                 Traceback (most recent call last)
<ipython-input-6-055d89daa74d> in <module>
     65 )
     66 
---> 67 trainer.train()
     68 
     69 

~/anaconda3/lib/python3.8/site-packages/transformers/trainer.py in train(self, resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)
   1258         model.zero_grad()
   1259 
-> 1260         self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
   1261 
   1262         # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.

~/anaconda3/lib/python3.8/site-packages/transformers/trainer_callback.py in on_train_begin(self, args, state, control)
    344     def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
    345         control.should_training_stop = False
--> 346         return self.call_event("on_train_begin", args, state, control)
    347 
    348     def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):

~/anaconda3/lib/python3.8/site-packages/transformers/trainer_callback.py in call_event(self, event, args, state, control, **kwargs)
    385     def call_event(self, event, args, state, control, **kwargs):
    386         for callback in self.callbacks:
--> 387             result = getattr(callback, event)(
    388                 args,
    389                 state,

~/anaconda3/lib/python3.8/site-packages/transformers/integrations.py in on_train_begin(self, args, state, control, model, **kwargs)
    538             self._initialized = False
    539         if not self._initialized:
--> 540             self.setup(args, state, model, **kwargs)
    541 
    542     def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):

~/anaconda3/lib/python3.8/site-packages/transformers/integrations.py in setup(self, args, state, model, **kwargs)
    511 
    512             if self._wandb.run is None:
--> 513                 self._wandb.init(
    514                     project=os.getenv("WANDB_PROJECT", "huggingface"),
    515                     name=run_name,

~/anaconda3/lib/python3.8/site-packages/wandb/sdk/wandb_init.py in init(job_type, dir, config, project, entity, reinit, tags, group, name, notes, magic, config_exclude_keys, config_include_keys, anonymous, mode, allow_val_change, resume, force, tensorboard, sync_tensorboard, monitor_gym, save_code, id, settings)
    906             if except_exit:
    907                 os._exit(-1)
--> 908             six.raise_from(Exception("problem"), error_seen)
    909     return run

~/anaconda3/lib/python3.8/site-packages/six.py in raise_from(value, from_value)

Exception: problem
```

The kernel didn't die this time, so I was able to try the training again, with the only correction being inclusion of previously inhibited duration clipper at 20s. My next experiments will be in reducing the per device batch size from 16 to 8. This also failed, so I'm down to 4.

The OSError[12] seems to be related to CPU. I noticed that only one core gets maxxed out, hinting that there is no parallelization in effect.