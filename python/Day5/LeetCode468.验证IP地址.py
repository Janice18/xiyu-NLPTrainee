class Solution(object):
    def validIPAddress(self, IP):
        """
        :type IP: str
        :rtype: str
        """
        if '.' in IP and self.checkIPv4(IP):
            return "IPv4"
        elif ':' in IP and self.checkIPv6(IP):
            return "IPv6"
        else:
            return "Neither"

    def checkIPv4(self, IP):
        numbers = IP.split('.')
        if len(numbers) != 4: return False
        for num in numbers:
            if not num  or (num[0] == '0' and len(num) != 1) or int(num) > 255:
                return False
        return True

    def checkIPv6(self, IP):
        IP = IP.lower()
        valid16 = "0123456789abcdef"
        if "::" in IP: return False
        numbers = IP.split(':')
        if len(numbers) != 8: return False
        for num in numbers:
            if not num: continue
            if len(num) >= 5: return False
            for n in num:
                if n not in valid16:
                    return False
        return True

if __name__ == '__main__':
    a1 = '172.16.254.1'
    a2 = '02001:0db8:85a3:0:0:8A2E:0370:7334'
    a3 = '256.256.256.256'
    a4 = '0201:0db8:85a3:0000:0000:8a2e:0370:7334'
    a5 = '02:234:123:0:2'
    print(Solution().validIPAddress(a1))
    print(Solution().validIPAddress(a2))
    print(Solution().validIPAddress(a3))
    print(Solution().validIPAddress(a4))
    print(Solution().validIPAddress(a5))





