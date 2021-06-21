import numpy as nmp
import wavio

from visual import visualize_track, show_spectrum_map

if __name__ == '__main__':
    tau = 0.5

    track: wavio.Wav = wavio.read("../dataset/custom_audio/track_1.wav")
    amplitude_max = max(track.data.max(), -track.data.min())
    data: nmp.ndarray = track.data / float(amplitude_max)
    visualize_track(data)
    data = data[:, 0]

    sample_size = int(track.rate * tau)
    sample_count = data.shape[0] // sample_size

    data = data[0:sample_size * sample_count].reshape((sample_count, sample_size))
    fft_data: nmp.ndarray = abs(nmp.fft.fft(data))
    fft_data = fft_data[:, 0:fft_data.shape[1] // 2].astype(int)
    spectrum_map = nmp.zeros((fft_data.shape[0], 3000), dtype=int)
    for i in range(fft_data.shape[0]):
        idx, values = nmp.unique(fft_data[i], return_counts=True)
        spectrum_map[i, idx] = values

    spectrum_map = nmp.transpose(spectrum_map[:, 68:])
    spectrum_map = spectrum_map / spectrum_map.max()

    show_spectrum_map(spectrum_map)
    # print(spectrum_map)
    # fft_res: nmp.ndarray = abs(nmp.fft.fft(data[:, 0]))
    # fft_res = fft_res[0:fft_res.shape[0] // 2 - 1]
    #
    # sample_size = int(track.rate * tau)
    # sample_count = fft_res.shape[0] // sample_size
    # fft_res = fft_res[0:sample_count * sample_size].reshape((sample_count, sample_size))
    # print(fft_res.shape)
