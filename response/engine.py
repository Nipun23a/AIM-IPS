class ResponseEngine:
    """
       Layer 3 - Dynamic response engine for AIM-IPS.
       Replaces the old dict-based ResponseEngine with full RequestContext support.
       Backward-compatible: also acceps old dict format via decide_lagancy().

    """

    def __init__(self,auto_blacklist : bool = True):
        """
           auto_blacklist: if True, automatically blacklist IPs on BLOCK decisions.
        """
        self.auto_blacklist = auto_blacklist
        self._redis = None 


    @property
    def redis(self):
        if self._redis is None:
            try:
                self._redis = get_redis()