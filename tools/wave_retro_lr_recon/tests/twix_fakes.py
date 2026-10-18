"""Synthetic mapVBVD IMAGE streams that reject bulk payload reads."""

import numpy as np


class FakeImageStream:
    """Expose coil-last test data through mapVBVD's named 16-axis interface."""

    def __init__(self, values, *, skip_lin=0, skip_par=0):
        """Wrap RO/LIN/PAR/[ECHO]/COIL values and record explicit read requests.

        Args:
            values: Complex test array on the stored grid.
            skip_lin: Logical origin of the stored LIN dimension.
            skip_par: Logical origin of the stored PAR dimension.
        """
        values = np.asarray(values, dtype=np.complex64)
        if values.ndim == 4:
            values = values[:, :, :, None, :]
        self.dataDims = ["Col", "Cha", "Lin", "Par", "Sli", "Ave", "Phs", "Eco",
                         "Rep", "Set", "Seg", "Ida", "Idb", "Idc", "Idd", "Ide"]
        ro, lin, par, echoes, coils = values.shape
        self.fullSize = np.array([ro, coils, lin, par, 1, 1, 1, echoes, *([1] * 8)])
        self.values = values.transpose(0, 4, 1, 2, 3).reshape(self.fullSize)
        self.skipLin, self.skipPar = skip_lin, skip_par
        self.flagRemoveOS = True
        self.squeeze = True
        self.average_dim = np.zeros(16, dtype=bool)
        self.selections = []

    def __getitem__(self, selection):
        """Return one unsqueezed PAR/ECHO block; reject bulk or OS-removed reads.

        Args:
            selection: Explicit index or slice for every named dimension.

        Returns:
            Selected array with all singleton dimensions retained.

        Raises:
            AssertionError: If a caller requests a bulk or ambiguous read.
        """
        assert isinstance(selection, tuple) and len(selection) == len(self.dataDims)
        assert not self.squeeze and not self.flagRemoveOS
        assert isinstance(selection[self.dataDims.index("Par")], int)
        assert isinstance(selection[self.dataDims.index("Eco")], int)
        self.selections.append(selection)
        slices = tuple(slice(index, index + 1) if isinstance(index, int) else index
                       for index in selection)
        return self.values[slices].copy()


class FakeRefscanStream(FakeImageStream):
    """Expose dense test calibration as selective REFSCAN reads only."""

    def __init__(self, values):
        """Wrap RO/LIN/PAR/SET/COIL data without allowing full-SET reads.

        Args:
            values: Dense synthetic calibration tensor with explicit SET axis.
        """
        values = np.asarray(values, dtype=np.complex64)
        ro, lin, par, sets, coils = values.shape
        super().__init__(np.zeros((ro, lin, par, coils), dtype=np.complex64))
        self.fullSize[self.dataDims.index("Set")] = sets
        self.values = values.transpose(0, 4, 1, 2, 3).reshape(self.fullSize)
        self.Set = np.arange(sets)

    def __getitem__(self, selection):
        """Read one explicit SET and PAR while recording all selected axes.

        Args:
            selection: Rank-preserving mapVBVD selections.

        Returns:
            Selected array with singleton dimensions retained.

        Raises:
            AssertionError: If any caller tries to load all SETs together.
        """
        assert isinstance(selection, tuple)
        assert isinstance(selection[self.dataDims.index("Set")], int)
        return super().__getitem__(selection)
