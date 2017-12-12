"""
Interface classes for state files.

"""
from datreant.core.backends.statefiles import TreantFile


class PlateMeasurementFile(TreantFile):

    def _init_state(self):
        super(PlateMeasurementFile, self)._init_state()
        self._state['plate_image'] = dict()
