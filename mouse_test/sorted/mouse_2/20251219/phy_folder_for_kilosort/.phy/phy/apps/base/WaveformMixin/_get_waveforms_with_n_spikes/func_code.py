# first line: 181
    def _get_waveforms_with_n_spikes(
            self, cluster_id, n_spikes_waveforms, current_filter=None):

        # HACK: we pass self.raw_data_filter.current_filter so that it is cached properly.
        pos = self.model.channel_positions

        # Only keep spikes from the spike waveforms selection.
        if self.model.spike_waveforms is not None:
            subset_spikes = self.model.spike_waveforms.spike_ids
            spike_ids = self.selector(
                n_spikes_waveforms, [cluster_id], subset_spikes=subset_spikes)
        # Or keep spikes from a subset of the chunks for performance reasons (decompression will
        # happen on the fly here).
        else:
            spike_ids = self.selector(n_spikes_waveforms, [cluster_id], subset_chunks=True)

        # Get the best channels.
        channel_ids = self.get_best_channels(cluster_id)
        channel_labels = self._get_channel_labels(channel_ids)

        # Load the waveforms, either from the raw data directly, or from the _phy_spikes* files.
        data = self.model.get_waveforms(spike_ids, channel_ids)
        if data is not None:
            data = data - np.median(data, axis=1)[:, np.newaxis, :]
        assert data.ndim == 3  # n_spikes, n_samples, n_channels

        # Filter the waveforms.
        if data is not None:
            data = self.raw_data_filter.apply(data, axis=1)
        return Bunch(
            data=data,
            channel_ids=channel_ids,
            channel_labels=channel_labels,
            channel_positions=pos[channel_ids],
        )
