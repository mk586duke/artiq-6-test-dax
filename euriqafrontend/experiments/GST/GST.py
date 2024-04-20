class Parser:
    def __init__(self, gate_set=["Gi", "Gx", "Gy"]):
        self.gate_set = gate_set
        pass

    def read(self, file_name):
        f = open(file_name, "r")
        f.readline()
        gst_set = []
        for l in f:
            seq = l.split()[0]
            gst = self.parse(seq)
            gst_set.append(gst)
        f.close()
        return gst_set

    def parse(self, seq):
        idx = 0
        gst = []
        while idx < len(seq):
            if seq[idx : idx + 2] == "{}":
                idx += 2
            elif seq[idx] == "(":
                idx += 1  # Skip '('
                # TODO: does not allow for nested () pairs.
                sub_seq = seq[idx:].split(")")[0]
                sub_gst = self.parse(sub_seq)
                idx += len(sub_seq) + 1  # +1 for closing ')'
                # Look for multiplier
                if idx < len(seq) and seq[idx] == "^":
                    idx += 1  # jump over '^'
                    num_str = ""
                    while idx < len(seq) and seq[idx].isdigit():
                        num_str += seq[idx]
                        idx += 1
                    for k in range(int(num_str)):
                        gst += sub_gst
                else:
                    gst += sub_gst
            else:
                # match longest key
                l = 0
                m = -1
                for k, g in enumerate(self.gate_set):
                    if len(g) > l and seq[idx : idx + len(g)] == g:
                        l = len(g)
                        m = k
                if m < 0:
                    raise (NameError("Unrecognized key"))
                idx += l
                gst.append([m])
        return gst


# x = read_GST(['Gi','Gx','Gy'])
# gst_set = x.read('MyDataTemplate.txt')
# print(gst_set)

# dill.source.getsource
