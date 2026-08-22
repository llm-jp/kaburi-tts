"""predictor = phone+SIL joint duration predictor for retrieval-free dialog TTS timing."""

PHONE_BIN_VALUES = list(range(1, 31))   # = 30 class, 1..30 frames
N_PHONE_CLASSES = len(PHONE_BIN_VALUES)

# Non-linear bin for SIL (= long-tail distribution)
SIL_BIN_VALUES = [
    0, 1, 2, 3, 4, 5,
    6, 8, 10, 12, 15, 20, 25, 30,
    40, 50, 65, 80, 100, 130, 160, 200, 250, 300,
]
N_SIL_CLASSES = len(SIL_BIN_VALUES)   # = 24

# Cross-channel turn-taking gap bins (= start_i - previous_any_end, in frames).
# Negative values represent overlap with the previous utterance on the other channel.
GAP_BIN_VALUES = list(range(-80, 205, 5))
N_GAP_CLASSES = len(GAP_BIN_VALUES)

# Token types (= used in TimingPredictorDataset)
TOKEN_TYPE_PAD = 0
TOKEN_TYPE_PHONE = 1
TOKEN_TYPE_PRE_SIL = 2
TOKEN_TYPE_FIRST_SIL = 3   # = first utt's pre_sil (= ignored in dur loss, but kept for accounting)
N_TOKEN_TYPES = 4

# Frame-level 4-state labels
STATE_NONE = 0     # = A silent, B silent
STATE_A_ONLY = 1   # = A active, B silent
STATE_B_ONLY = 2   # = A silent, B active
STATE_BOTH = 3     # = A active, B active
N_STATES = 4
