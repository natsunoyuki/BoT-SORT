"""
BoT-SORT tracking algorithm originally written by NirAharon
https://github.com/NirAharon/BoT-SORT

Cleaned up and refactored for something more user friendly and deployable
in a production setting.
"""
from pathlib import Path
from typing import List, Tuple, Union, Dict

import numpy as np
from numpy.typing import NDArray

from bot_sort import matching
from bot_sort.gmc import GMC
from bot_sort.s_track import STrack
from bot_sort.basetrack import BaseTrack, TrackState
from bot_sort.kalman_filter import KalmanFilter

from bot_sort.reid.fast_reid_interface import FastReIDInterface


# %% BoTSORT - BoT-SORT tracker object.
class BoTSORT(object):
    def __init__(
        self, 
        # SORT track (without Re-ID or camera motion correction):
        track_high_thresh: float = 0.6,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.7,
        match_thresh: float = 0.8,
        track_buffer: float = 30,
        frame_rate: float = 30,
        # ReID:
        with_reid: bool = False,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        # TODO FastReID:
        fast_reid_config: Path = Path("fast_reid/configs/MOT17/sbs_S50.yml"),
        fast_reid_weights: Path = Path("pretrained/mot17_sbs_S50.pth"),
        fast_reid_device: str = "cpu",
        # TODO Camera motion correction:
        with_gmc: bool = False,
        cmc_method: str = "file",
        cmc_name: str = "cmc_name",
        cmc_ablation: bool = False,
    ):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # ReID.
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        if self.with_reid:
            self.encoder = FastReIDInterface(
                fast_reid_config, fast_reid_weights, fast_reid_device
            )
        else:
            self.encoder = None

        # Camera motion correction.
        if with_gmc:
            self.gmc = GMC(method=cmc_method, verbose=[cmc_name, cmc_ablation])
        else:
            self.gmc = None


    def update(
        self, 
        bboxes: Union[List, NDArray] = [], 
        classes: Union[List, NDArray] = [], 
        scores: Union[List, NDArray] = [], 
        # TODO img for re-id.
        img = None,
    ) -> List:
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(bboxes) > 0:
            assert len(bboxes) == len(scores) == len(classes)
            
            # Remove bad detections.
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold (high confidence) detections.
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings for re-ID'''
        if self.with_reid:
            features_keep = self.encoder.inference(img, dets)

        '''Detections'''
        if len(dets) > 0:
            if self.with_reid:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, c, f) for
                    (tlbr, s, c, f) in zip(
                        dets, scores_keep, classes_keep, features_keep)
                ]
            else:
                detections = [
                    STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                    (tlbr, s, c) in zip(dets, scores_keep, classes_keep)
                ]
        else:
            detections = []

        '''Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        '''Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with the Kalman Filter.
        STrack.multi_predict(strack_pool)

        # Fix camera motion.
        if self.gmc is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes.
        ious_dists = matching.iou_distance(strack_pool, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        ious_dists = matching.fuse_score(ious_dists, detections)

        if self.with_reid:
            # TODO this method of combining the IoU and encoding distances 
            # c.f. equation 13 in https://arxiv.org/pdf/2206.14651
            # has been proven in some use cases to be unreliable. 
            # A modification is required to ensure greater tracking reliability.
            emb_dists = matching.embedding_distance(strack_pool, detections) / 2
            # raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(
            #     strack_pool, detections)
            # dists = matching.fuse_motion(
            #     self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, c) for
                (tlbr, s, c) in zip(dets_second, scores_second, classes_second)]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i] 
            for i in u_track if strack_pool[i].state == TrackState.Tracked]
        
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning 
        frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        ious_dists = matching.fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections) / 2
            # raw_emb_dists = emb_dists.copy()
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """Merge"""
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        # output_stracks = [
        #     track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
