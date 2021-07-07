# This is a preprocessing sample of NSL-KDD Train dataset
# Main approach is onehot encoding
import numpy as np
from sklearn.preprocessing import MinMaxScaler
class Preprocessing:
    def __init__(self, path):
        self.Class = {'Dos': ['back', 'neptune', 'smurf', 'teardrop', 'land', 'pod', 'apache2', 'mailbomb', 'processtable'],
                 'Probe': ['satan', 'portsweep', 'ipsweep', 'nmap', 'mscan', 'sain'],
                 'R2L': ['warezmaster', 'warezclient', 'ftpwrite', 'guesspassword', 'imap', 'multihop', 'phf', 'spy',
                         'sendmail', 'named', 'snmpgetattack', 'snmpguess', 'xlock', 'xsnoop', 'worm'],
                 'U2R': ['rootkit', 'bufferoverflow', 'loadmodule', 'perl']}

        self.protocol = ['tcp', 'udp', 'icmp']

        self.service = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                   'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                   'hostnames', 'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin',
                   'kshell', 'ldap', 'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn',
                   'netstat', 'nnsp', 'nntp', 'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private',
                   'red_i', 'remote_job', 'rje', 'shell', 'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat',
                   'telnet', 'tftp_u', 'tim_i', 'time', 'urh_i', 'urp_i', 'uucp', 'uucp_path', 'vmnet', 'whois', 'X11',
                   'Z39_50']
        self.flag = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
        self.path = path
        self.DOS, self.PROBE, self.R2L, self.U2R, self.Normal = 0, 0, 0, 0, 0
        self.data, self.label = None, None
        self.maxmin = MinMaxScaler()

    def onehot_encoder(self, code, mode):
        if mode == 'protocol':
            one_hot = np.zeros(len(self.protocol))
            one_hot[self.protocol.index(code)] = 1
        elif mode == 'service':
            one_hot = np.zeros(len(self.service))
            one_hot[self.service.index(code)] = 1
        else:
            one_hot = np.zeros(len(self.flag))
            one_hot[self.flag.index(code)] = 1
        return one_hot

    def readfile(self):
        with open(self.path) as f:
            lines = f.readlines()
        return lines

    def deal_with_lines(self):
        lines = self.readfile()
        self.data = np.zeros((len(lines), 122))

        self.label = np.zeros((len(lines)))
        for idx, line in enumerate(lines):
            line = line.split(',')
            if line[-2] in self.Class['Dos']:
                label = 1.
                self.DOS += 1
            elif line[-2] in self.Class['Probe']:
                label = 2.
                self.PROBE += 1
            elif line[-2] in self.Class['R2L']:
                label = 3.
                self.R2L += 1
            elif line[-2] in self.Class['U2R']:
                label = 4.
                self.U2R += 1
            else:
                label = 0.
                self.Normal += 1
            line.remove(line[-2])
            line.remove(line[-1])

            pro = self.onehot_encoder(line[1], 'protocol')
            ser = self.onehot_encoder(line[2], 'service')
            other = self.onehot_encoder(line[3], 'others')
            line.remove(line[1])
            line.remove(line[1])
            line.remove(line[1])
            line.extend(pro)
            line.extend(ser)
            line.extend(other)
            line = list(map(float, line))
            line = np.array(line)
            self.data[idx, :] = line
            self.label[idx] = label
        return self.maxmin.fit_transform(self.data), self.label

if __name__ == '__main__':
    Pre = Preprocessing('KDDTrain+.txt')
    train_data, train_label = Pre.deal_with_lines()
    print(train_data.shape, train_label.shape)
    #(125973, 122) (125973,)
