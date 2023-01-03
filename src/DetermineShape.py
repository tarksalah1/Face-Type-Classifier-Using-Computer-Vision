class DetermineShape:
    def __init__(self):
        self.shape = "Not detected"

    def deterShape(self, rdic):
        R1 = rdic["R1"]
        R2 = rdic["R2"]
        R3 = rdic["R3"]
        R4 = rdic["R4"]
        R5 = rdic["R5"]
        R6 = rdic["R6"]
        R7 = rdic["R7"]
        R8 = rdic["R8"]
        R9 = rdic["R9"]
        R10 = rdic["R10"]
        A1 = rdic["A1"]
        A2 = rdic["A2"]
        A3 = rdic["A3"]
        if R2 > 72.31:
            if R2 <= 77.87:
                if R4 <= 116.94:
                    if R2 <= 73.78:
                        if R8 <= 95.69:
                            self.shape = "square"
                        else:
                            self.shape = "oblong or oval"
                    else:
                        if R4 <= 115:
                            self.shape = "square"
                        else:
                            self.shape = "oval"
                else:
                    if R3 <= 67.81:
                        if A2 <= 45:
                            self.shape = "oval or oblong"
                        else:
                            self.shape = "oval"
                    else:
                        if A2 < 55:
                            self.shape = "heart or oval"
                        else:
                            self.shape = "oval or heart"
            else:
                if R7 <= 50.48:
                    if R5 <= 57.71:
                        if R2 <= 80.56:
                            if R4 < 117:
                                self.shape = "square or round"
                            else:
                                self.shape = "round or heart"
                        else:
                            self.shape = "round or heart"
                    else:
                        self.shape = "square or Heart"
                else:
                    self.shape = "square"
        else:
            self.shape = "oblong "


        return self.shape
