from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips


class Mice(VisionDataset):
    def __init__(self, root, frames_per_clip, step_between_clips=1, frame_rate=None,
                 extensions=("mp4",), transform=None, _precomputed_metadata=None,
                 num_workers=1, _video_width=0, _video_height=0,
                 _video_min_dimension=0, _audio_samples=0, _audio_channels=0):
        super(Mice, self).__init__(root)
        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]

        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            _audio_channels=_audio_channels,
        )
        self.transform = transform

    @property
    def metadata(self):
        return self.video_clips.metadata

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        video, _, _, video_idx = self.video_clips.get_clip(idx)
        video_idx, clip_idx = self.video_clips.get_clip_location(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label, video_idx, clip_idx
    #
    #
    # def __getitem__(self, idx):
    #     video, _, info, video_idx = self.video_clips.get_clip(idx)
    #     video_idx, clip_idx = self.video_clips.get_clip_location(idx)
    #     label = self.samples[video_idx][1]
    #     if self.transform is not None:
    #         video = self.transform(video)
    #     return video, label, video_idx, clip_idx