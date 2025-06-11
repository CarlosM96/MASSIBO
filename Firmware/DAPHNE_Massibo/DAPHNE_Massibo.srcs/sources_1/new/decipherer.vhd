-------------------------------------------------------------------------------
--
-- Title       : Decipherer
-- Design      : ethernet_controller
-- Author      : Ryan Rivera
-- Company     : FNAL
--
-------------------------------------------------------------------------------
--
-- File        : d:\Projects\otsdaq\OtS Ethernet MAC firmware\ActiveHDL_proj\ethernet_controller\compile\decipherer.vhd
-- Generated   : 02/09/16 15:17:28
-- From        : d:/Projects/otsdaq/OtS Ethernet MAC firmware/ActiveHDL_proj/ethernet_controller/src/decipherer.asf
-- By          : FSM2VHDL ver. 5.0.7.2
--
-------------------------------------------------------------------------------
--
-- Description : 
--
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity decipherer is 
	port (
		clk: in STD_LOGIC;
		data_in: in STD_LOGIC_VECTOR (7 downto 0);
		dv: in STD_LOGIC;
		er: in STD_LOGIC;
		reset: in STD_LOGIC;
		self_addrs: in STD_LOGIC_VECTOR (31 downto 0);
		arp_req_ip: out STD_LOGIC_VECTOR (31 downto 0);
		arp_req_mac: out STD_LOGIC_VECTOR (47 downto 0);
		arp_search_ip: out STD_LOGIC_VECTOR (31 downto 0);
		capture_source_addrs: out STD_LOGIC;
		clken_out: out STD_LOGIC;
		crc_chk_en: out STD_LOGIC;
		crc_chk_init: out STD_LOGIC;
		crc_chk_rd: out STD_LOGIC;
		data_out: out STD_LOGIC_VECTOR (7 downto 0);
		dest_mac: out STD_LOGIC_VECTOR (47 downto 0);
		four_bit_mode_out: out STD_LOGIC;
		icmp_checksum: out STD_LOGIC_VECTOR (15 downto 0);
		ip_data_count: out STD_LOGIC_VECTOR (10 downto 0);
		is_arp: out STD_LOGIC;
		is_icmp_ping: out STD_LOGIC;
		is_idle: out STD_LOGIC;
		is_ip: out STD_LOGIC;
		is_udp: out STD_LOGIC;
		src_mac: out STD_LOGIC_VECTOR (47 downto 0);
		udp_data_count: out STD_LOGIC_VECTOR (10 downto 0);
		udp_data_valid: out STD_LOGIC;
		udp_dest_port_out: out STD_LOGIC_VECTOR (15 downto 0);
		udp_src_ip: out STD_LOGIC_VECTOR (31 downto 0);
		udp_src_port: out STD_LOGIC_VECTOR (15 downto 0));
end decipherer;

architecture decipherer_arch of decipherer is

-- diagram signals declarations
signal addrs_match_sig: STD_LOGIC;
signal capture_source_addrs_sig: STD_LOGIC;
signal clken: STD_LOGIC;
signal crc_chk_en_masked: STD_LOGIC;
signal crc_chk_en_unmasked: STD_LOGIC;
signal data: STD_LOGIC_VECTOR (7 downto 0);
signal dv_old: STD_LOGIC;
signal first_bytes_count: STD_LOGIC_VECTOR (2 downto 0);
signal four_bit_count: STD_LOGIC_VECTOR (4 downto 0);
signal four_bit_data: STD_LOGIC_VECTOR (7 downto 0);
signal four_bit_mode: STD_LOGIC;
signal icmp_trigger_sig: STD_LOGIC;
signal is_arp_sig: STD_LOGIC;
signal is_icmp_ping_sig: STD_LOGIC;
signal is_ip_sig: STD_LOGIC;
signal is_udp_sig: STD_LOGIC;
signal udp_countdown: STD_LOGIC_VECTOR (15 downto 0);
signal udp_data_valid_sig: STD_LOGIC;
signal udp_dest_ip: STD_LOGIC_VECTOR (31 downto 0);
signal udp_dest_port: STD_LOGIC_VECTOR (15 downto 0);
signal udp_zeros: STD_LOGIC_VECTOR (10 downto 0);

-- SYMBOLIC ENCODED state machine: Sreg0
type Sreg0_type is (
    Idle, RecvPacket_IP_Payload_UDP_RecvDataLoop, RecvPacket_Type_S29, RecvPacket_IP_Payload_IP_TotLength2, RecvPacket_IP_Payload_UDP_SourcePort1,
    RecvPacket_IP_Payload_IP_VersionAndHeader, RecvPacket_Dest_S22, RecvPacket_IP_Payload_IP_TotLength1, RecvPacket_IP_Payload_IP_ToS,
    RecvPacket_Dest_S11, RecvPacket_IP_Payload_UDP_DestPort1, RecvPacket_IP_Payload_IP_TTL, RecvPacket_IP_Payload_IP_FragmentOffset,
    RecvPacket_IP_Payload_IP_FlagsAndFrag, RecvPacket_IP_Payload_IP_ID2, RecvPacket_IP_Payload_IP_ID1, RecvPacket_Dest_S12, RecvPacket_Dest_S13,
    RecvPacket_Dest_S14, RecvPacket_Dest_S15, RecvPacket_IP_Payload_IP_SourceAddr1, RecvPacket_IP_Payload_IP_Checksum2, RecvPacket_IP_Payload_IP_Checksum1,
    RecvPacket_IP_Payload_IP_Protocol, RecvPacket_IP_Payload_IP_SourceAddr3, RecvPacket_IP_Payload_IP_SourceAddr2, RecvPacket_IP_Payload_IP_DestAddr2,
    RecvPacket_IP_Payload_IP_DestAddr1, RecvPacket_IP_Payload_IP_SourceAddr4, RecvPacket_IP_Payload_IP_DestAddr4, RecvPacket_IP_Payload_IP_DestAddr3,
    RecvPacket_Type_S26, RecvPacket_IP_Payload_UDP_DestPort2, RecvPacket_IP_Payload_UDP_Length1, RecvPacket_IP_Payload_UDP_Checksum2,
    RecvPacket_IP_Payload_UDP_Checksum1, RecvPacket_IP_Payload_UDP_Length2, RecvPacket_IP_Payload_UDP_SourcePort2, RecvPacket_Src_S42,
    RecvPacket_Src_S44, RecvPacket_Src_S45, RecvPacket_Src_S46, RecvPacket_Src_S47, RecvPacket_Src_S43, RecvPacket_Type_S1, RecvPacket_Type_S48,
    RecvPacket_Type_S49, RecvPacket_ARP_Payload_HType2, RecvPacket_ARP_Payload_HType1, RecvPacket_ARP_Payload_PType1, RecvPacket_ARP_Payload_PType2,
    RecvPacket_ARP_Payload_HLen, RecvPacket_ARP_Payload_PLen, RecvPacket_ARP_Payload_Op1, RecvPacket_ARP_Payload_SMac1, RecvPacket_ARP_Payload_SMac2,
    RecvPacket_ARP_Payload_SMac3, RecvPacket_ARP_Payload_SMac4, RecvPacket_ARP_Payload_SMac5, RecvPacket_ARP_Payload_SMac6, RecvPacket_ARP_Payload_Sip1,
    RecvPacket_ARP_Payload_Sip2, RecvPacket_ARP_Payload_Sip3, RecvPacket_ARP_Payload_Sip4, RecvPacket_ARP_Payload_TMac1, RecvPacket_ARP_Payload_TMac2,
    RecvPacket_ARP_Payload_TMac3, RecvPacket_ARP_Payload_TMac4, RecvPacket_ARP_Payload_TMac5, RecvPacket_ARP_Payload_TMac6, RecvPacket_ARP_Payload_Tip1,
    RecvPacket_ARP_Payload_Tip2, RecvPacket_ARP_Payload_Tip3, RecvPacket_ARP_Payload_Tip4, RecvPacket_IP_Payload_ICMP_ID1, RecvPacket_ARP_Payload_Op2,
    RecvPacket_IP_Payload_ICMP_ID2, RecvPacket_Preamble_S50, RecvPacket_CRC_ARP_S52, RecvPacket_IP_Payload_ICMP_SeqNum1, RecvPacket_CRC_ARP_S53,
    RecvPacket_CRC_ARP_crc1, RecvPacket_IP_Payload_ICMP_SeqNum2, RecvPacket_CRC_ARP_crc2, RecvPacket_CRC_ARP_crc3, RecvPacket_CRC_ARP_crc4,
    RecvPacket_IP_Payload_ICMP_DataLoop, RecvPacket_Preamble_S54, RecvPacket_CRC_IP_S55, RecvPacket_CRC_IP_crc6, RecvPacket_CRC_IP_crc7,
    RecvPacket_CRC_IP_crc8, RecvPacket_CRC_IP_crc9, RecvPacket_CRC_IP_S56, RecvPacket_Preamble_S57, Ready, RecvPacket_IP_Payload_ICMP_Type,
    RecvPacket_IP_Payload_ICMP_Code, RecvPacket_IP_Payload_ICMP_Checksum1, RecvPacket_IP_Payload_ICMP_Checksum2
);
-- attribute ENUM_ENCODING of Sreg0_type: type is ... -- enum_encoding attribute is not supported for symbolic encoding

signal Sreg0: Sreg0_type;

begin

-- concurrent signals assignments

-- Diagram ACTION
is_arp <= is_arp_sig;
is_ip <= is_ip_sig;
is_udp <= is_udp_sig;
is_icmp_ping <= icmp_trigger_sig and addrs_match_sig;
-- FIXED to consider icmp type and dest address.. old: is_icmp_ping_sig;
capture_source_addrs <= capture_source_addrs_sig and addrs_match_sig;
clken_out <= clken;
data_out <= data;
data <=  data_in when (four_bit_mode = '0') else four_bit_data;
crc_chk_en_masked <= crc_chk_en_unmasked and clken and dv_old;
crc_chk_en <=  (crc_chk_en_unmasked and dv) when four_bit_mode = '0' else crc_chk_en_masked;
udp_data_valid <= udp_data_valid_sig and addrs_match_sig;
-- detect 4 bit interface (100 Mbps)
-- using first few bytes
four_bit_proc : process(clk)
begin
	if rising_edge(clk) then
		dv_old <= dv;
		if (Sreg0 = Ready) then
			four_bit_count <= (others => '0');
			first_bytes_count <= (others => '0');
			four_bit_mode <= '0';
			clken <= '1';
		elsif (four_bit_mode = '0' and first_bytes_count < 5) then
			clken <= '1';
			first_bytes_count <= first_bytes_count + 1;
			--count total bytes
			if (data_in(7 downto 4) = 0) then  --count 4 bit bytes
				four_bit_count <= four_bit_count + 1;
			end if;
			if(four_bit_count > 3) then -- 4 bit mode detected!
				four_bit_mode <= '1';
				four_bit_count <= (others => '0');
			end if;
		elsif (four_bit_mode = '1') then
			if (four_bit_count < 9) then  -- let data catch up to state machine by withholding clken
				four_bit_count <= four_bit_count + 1;
				clken <= '0';
			else
				clken <= not clken;
			end if;
			if(clken = '1') then
				four_bit_data(3 downto 0) <= data_in(3 downto 0);
			else
				four_bit_data(7 downto 4) <= data_in(3 downto 0);
			end if;
		else	-- 8bit data mode
			clken <= '1';
		end if;
	end if;
end process;
-- get received data size and calc number of 0 bytess to fill
udp_dc : process(clk)
begin
if rising_edge(clk) then
	if reset = '1' then
		udp_data_count <= (others => '0');
		udp_zeros <= (others => '0');
	elsif Sreg0 = recvpacket_ip_payload_ip_checksum1 then		 -- ip length for icmp
		ip_data_count <= udp_countdown(10 downto 0) - ("000" & x"08");
	elsif Sreg0 = recvpacket_ip_payload_udp_length2 then		 -- udp length for rx output
		udp_data_count <= udp_countdown(10 downto 0) - ("000" & x"08");
-- if number of bytes < 18 then need to add 0's	(header is 8)
		if udp_countdown(10 downto 0) < ("000" & x"1A")	then
			udp_zeros <= ("000" & x"1A") - udp_countdown(10 downto 0);
		else
			udp_zeros <= (others => '0');
		end if;
	end if;
end if;
end process;
udp_dest_port_out <= udp_dest_port;
-- Change Feb 2016 (no longer assume first 3 bytes of IP)
-- NOTE: Only IP is matched. Any mac and port are accepted
match_proc : process(clk)
begin
if rising_edge(clk) then
	addrs_match_sig <= '0';
	if (udp_dest_ip = self_addrs) then -- and
--(self_port = 0 or udp_dest_port = self_port)) then
-- Note: rejecting the port presented a problem for ICMP matching logic
--(x"C0A885" & addrs) then --this UDP packet was intended for this firmware.
-- Removed feature: -- or udp_dest_ip = x"C0A885FE" then --0xFE is CAPTAN broadcast
-- Note: this is not considering the mac address (shouldn't matter if ARP works?)
		addrs_match_sig <= '1';
	end if;
end if;
end process;

----------------------------------------------------------------------
-- Machine: Sreg0
----------------------------------------------------------------------
Sreg0_machine: process (clk)
begin
	if clk'event and clk = '1' then
		if reset = '1' then
			Sreg0 <= Idle;
			-- Set default values for outputs, signals and variables
			-- ...
-- **** only clear these on reset ****
--from ethernet frame
			dest_mac <= (others => '0');
			src_mac <= (others => '0');
--from arp payload
			arp_req_mac <= (others => '0');
			arp_req_ip <= (others => '0');
			arp_search_ip <= (others => '0');
--from ipv4 payload
			udp_src_ip <= (others => '0');
			udp_src_port <= (others => '0');
			udp_dest_ip <= (others => '0');
			four_bit_mode_out <= '0';
			is_ip_sig <= '0';
			is_arp_sig <= '0';
			is_idle <= '1';
			is_udp_sig <= '0';
			is_icmp_ping_sig <= '0';
			icmp_trigger_sig <= '0';
--from ipv4 payload
			udp_data_valid_sig <= '0';
			--indicates packet data on data lines
			udp_countdown <= (others => '0');
			-- used to determine when udp packet data ends
--also used for various delay counts throughout reception
			crc_chk_init <= '0';
			crc_chk_en_unmasked <= '0';
			crc_chk_rd <= '0';
			capture_source_addrs_sig <= '0';
		else
			if clken = '1' then
				-- Set default values for outputs, signals and variables
				-- ...
				case Sreg0 is
					when Idle =>
						is_ip_sig <= '0';
						is_arp_sig <= '0';
						is_idle <= '1';
						is_udp_sig <= '0';
						is_icmp_ping_sig <= '0';
						icmp_trigger_sig <= '0';
						--from ipv4 payload
						udp_data_valid_sig <= '0';
						--indicates packet data on data lines
						udp_countdown <= (others => '0');
						-- used to determine when udp packet data ends
						--also used for various delay counts throughout reception
						crc_chk_init <= '0';
						crc_chk_en_unmasked <= '0';
						crc_chk_rd <= '0';
						capture_source_addrs_sig <= '0';
						if dv = '0' and er ='0' then
							Sreg0 <= Ready;
							udp_dest_ip <= (others => '0');
							-- reset for checking destination
						end if;
					when Ready =>
						if dv = '1' and er = '0' then
							Sreg0 <= RecvPacket_Preamble_S50;
							is_idle <= '0';
							crc_chk_init <= '1';
							-- reset crc calculation
						end if;
					when RecvPacket_IP_Payload_UDP_RecvDataLoop =>
						udp_countdown <= udp_countdown - 1;
						if udp_countdown = x"0009" then
							Sreg0 <= RecvPacket_CRC_IP_S55;
							udp_data_valid_sig <= '0';
						end if;
					when RecvPacket_IP_Payload_UDP_SourcePort1 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_SourcePort2;
						udp_src_port(7 downto 0) <= data;
						-- acquire src port to be used as destination port from CAPTAN
					when RecvPacket_IP_Payload_UDP_DestPort1 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_DestPort2;
						udp_dest_port(7 downto 0) <= data;
						--acquire dest port for possible forwarding
					when RecvPacket_IP_Payload_UDP_DestPort2 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_Length1;
						udp_countdown(15 downto 8) <= data;
					when RecvPacket_IP_Payload_UDP_Length1 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_Length2;
						udp_countdown(7 downto 0) <= data;
					when RecvPacket_IP_Payload_UDP_Checksum2 =>
						if udp_countdown <= x"0009" then	--  for the 1 byte case (or illegal case)
							Sreg0 <= RecvPacket_CRC_IP_S55;
							udp_data_valid_sig <= '0';
						else
							Sreg0 <= RecvPacket_IP_Payload_UDP_RecvDataLoop;
							udp_countdown <= udp_countdown - 1;
						end if;
					when RecvPacket_IP_Payload_UDP_Checksum1 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_Checksum2;
						if udp_countdown > 9 then
							udp_data_valid_sig <= '1';
						elsif udp_countdown = 9 then -- is 1 byte packet (set IP address)
							capture_source_addrs_sig <= '1';
						end if;
					when RecvPacket_IP_Payload_UDP_Length2 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_Checksum1;
					when RecvPacket_IP_Payload_UDP_SourcePort2 =>
						Sreg0 <= RecvPacket_IP_Payload_UDP_DestPort1;
						udp_dest_port(15 downto 8) <= data;
						--acquire dest port for possible forwarding
					when RecvPacket_IP_Payload_IP_TotLength2 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_ID1;
					when RecvPacket_IP_Payload_IP_VersionAndHeader =>
						Sreg0 <= RecvPacket_IP_Payload_IP_ToS;
					when RecvPacket_IP_Payload_IP_TotLength1 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_TotLength2;
						udp_countdown(7 downto 0) <= data;
					when RecvPacket_IP_Payload_IP_ToS =>
						Sreg0 <= RecvPacket_IP_Payload_IP_TotLength1;
						udp_countdown(15 downto 8) <= data;
					when RecvPacket_IP_Payload_IP_TTL =>
						if data = x"11" then
							Sreg0 <= RecvPacket_IP_Payload_IP_Protocol;
							is_udp_sig <= '1';
						elsif data = x"01" then
							Sreg0 <= RecvPacket_IP_Payload_IP_Protocol;
							is_icmp_ping_sig <= '1';
						else
							Sreg0 <= Idle;
						end if;
					when RecvPacket_IP_Payload_IP_FragmentOffset =>
						Sreg0 <= RecvPacket_IP_Payload_IP_TTL;
					when RecvPacket_IP_Payload_IP_FlagsAndFrag =>
						Sreg0 <= RecvPacket_IP_Payload_IP_FragmentOffset;
					when RecvPacket_IP_Payload_IP_ID2 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_FlagsAndFrag;
					when RecvPacket_IP_Payload_IP_ID1 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_ID2;
					when RecvPacket_IP_Payload_IP_SourceAddr1 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_SourceAddr2;
						udp_src_ip(23 downto 16) <= data;
					when RecvPacket_IP_Payload_IP_Checksum2 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_SourceAddr1;
						udp_src_ip(31 downto 24) <= data;
					when RecvPacket_IP_Payload_IP_Checksum1 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_Checksum2;
					when RecvPacket_IP_Payload_IP_Protocol =>
						Sreg0 <= RecvPacket_IP_Payload_IP_Checksum1;
					when RecvPacket_IP_Payload_IP_SourceAddr3 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_SourceAddr4;
						udp_src_ip(7 downto 0) <= data;
					when RecvPacket_IP_Payload_IP_SourceAddr2 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_SourceAddr3;
						udp_src_ip(15 downto 8) <= data;
					when RecvPacket_IP_Payload_IP_DestAddr2 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_DestAddr3;
						udp_dest_ip(15 downto 8) <= data;
					when RecvPacket_IP_Payload_IP_DestAddr1 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_DestAddr2;
						udp_dest_ip(23 downto 16) <= data;
					when RecvPacket_IP_Payload_IP_SourceAddr4 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_DestAddr1;
						udp_dest_ip(31 downto 24) <= data;
					when RecvPacket_IP_Payload_IP_DestAddr4 =>
						if (is_icmp_ping_sig = '1') and (data = x"08") then	-- ping echo request
							Sreg0 <= RecvPacket_IP_Payload_ICMP_Type;
							udp_countdown <= udp_countdown - 28;
							-- get ping payload data length
							icmp_trigger_sig <= '1';
							-- trigger packet create
						elsif is_icmp_ping_sig = '1' then
							Sreg0 <= Idle;
						elsif is_udp_sig = '1' then
							Sreg0 <= RecvPacket_IP_Payload_UDP_SourcePort1;
							udp_src_port(15 downto 8) <= data;
							-- acquire src port to be used as destination port from CAPTAN
						end if;
					when RecvPacket_IP_Payload_IP_DestAddr3 =>
						Sreg0 <= RecvPacket_IP_Payload_IP_DestAddr4;
						udp_dest_ip(7 downto 0) <= data;
					when RecvPacket_IP_Payload_ICMP_ID1 =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_ID2;
					when RecvPacket_IP_Payload_ICMP_ID2 =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_SeqNum1;
					when RecvPacket_IP_Payload_ICMP_SeqNum1 =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_SeqNum2;
					when RecvPacket_IP_Payload_ICMP_SeqNum2 =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_DataLoop;
					when RecvPacket_IP_Payload_ICMP_DataLoop =>
						udp_countdown <= udp_countdown - 1;
						if udp_countdown = 1 then
							Sreg0 <= RecvPacket_CRC_IP_S55;
						end if;
					when RecvPacket_IP_Payload_ICMP_Type =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_Code;
					when RecvPacket_IP_Payload_ICMP_Code =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_Checksum1;
						icmp_checksum(15 downto 8) <= data;
					when RecvPacket_IP_Payload_ICMP_Checksum1 =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_Checksum2;
						icmp_checksum(7 downto 0) <= data;
					when RecvPacket_IP_Payload_ICMP_Checksum2 =>
						Sreg0 <= RecvPacket_IP_Payload_ICMP_ID1;
					when RecvPacket_ARP_Payload_HType2 =>
						Sreg0 <= RecvPacket_ARP_Payload_PType1;
					when RecvPacket_ARP_Payload_HType1 =>
						Sreg0 <= RecvPacket_ARP_Payload_HType2;
					when RecvPacket_ARP_Payload_PType1 =>
						Sreg0 <= RecvPacket_ARP_Payload_PType2;
					when RecvPacket_ARP_Payload_PType2 =>
						Sreg0 <= RecvPacket_ARP_Payload_HLen;
					when RecvPacket_ARP_Payload_HLen =>
						Sreg0 <= RecvPacket_ARP_Payload_PLen;
					when RecvPacket_ARP_Payload_PLen =>
						Sreg0 <= RecvPacket_ARP_Payload_Op1;
					when RecvPacket_ARP_Payload_Op1 =>
						Sreg0 <= RecvPacket_ARP_Payload_Op2;
					when RecvPacket_ARP_Payload_SMac1 =>
						Sreg0 <= RecvPacket_ARP_Payload_SMac2;
						arp_req_mac(39 downto 32) <= data;
					when RecvPacket_ARP_Payload_SMac2 =>
						Sreg0 <= RecvPacket_ARP_Payload_SMac3;
						arp_req_mac(31 downto 24) <= data;
					when RecvPacket_ARP_Payload_SMac3 =>
						Sreg0 <= RecvPacket_ARP_Payload_SMac4;
						arp_req_mac(23 downto 16) <= data;
					when RecvPacket_ARP_Payload_SMac4 =>
						Sreg0 <= RecvPacket_ARP_Payload_SMac5;
						arp_req_mac(15 downto 8) <= data;
					when RecvPacket_ARP_Payload_SMac5 =>
						Sreg0 <= RecvPacket_ARP_Payload_SMac6;
						arp_req_mac(7 downto 0) <= data;
					when RecvPacket_ARP_Payload_SMac6 =>
						Sreg0 <= RecvPacket_ARP_Payload_Sip1;
						arp_req_ip(31 downto 24) <= data;
					when RecvPacket_ARP_Payload_Sip1 =>
						Sreg0 <= RecvPacket_ARP_Payload_Sip2;
						arp_req_ip(23 downto 16) <= data;
					when RecvPacket_ARP_Payload_Sip2 =>
						Sreg0 <= RecvPacket_ARP_Payload_Sip3;
						arp_req_ip(15 downto 8) <= data;
					when RecvPacket_ARP_Payload_Sip3 =>
						Sreg0 <= RecvPacket_ARP_Payload_Sip4;
						arp_req_ip(7 downto 0) <= data;
					when RecvPacket_ARP_Payload_Sip4 =>
						Sreg0 <= RecvPacket_ARP_Payload_TMac1;
					when RecvPacket_ARP_Payload_TMac1 =>
						Sreg0 <= RecvPacket_ARP_Payload_TMac2;
					when RecvPacket_ARP_Payload_TMac2 =>
						Sreg0 <= RecvPacket_ARP_Payload_TMac3;
					when RecvPacket_ARP_Payload_TMac3 =>
						Sreg0 <= RecvPacket_ARP_Payload_TMac4;
					when RecvPacket_ARP_Payload_TMac4 =>
						Sreg0 <= RecvPacket_ARP_Payload_TMac5;
					when RecvPacket_ARP_Payload_TMac5 =>
						Sreg0 <= RecvPacket_ARP_Payload_TMac6;
					when RecvPacket_ARP_Payload_TMac6 =>
						Sreg0 <= RecvPacket_ARP_Payload_Tip1;
						arp_search_ip(31 downto 24) <= data;
					when RecvPacket_ARP_Payload_Tip1 =>
						Sreg0 <= RecvPacket_ARP_Payload_Tip2;
						arp_search_ip(23 downto 16) <= data;
					when RecvPacket_ARP_Payload_Tip2 =>
						Sreg0 <= RecvPacket_ARP_Payload_Tip3;
						arp_search_ip(15 downto 8) <= data;
					when RecvPacket_ARP_Payload_Tip3 =>
						Sreg0 <= RecvPacket_ARP_Payload_Tip4;
						arp_search_ip(7 downto 0) <= data;
					when RecvPacket_ARP_Payload_Tip4 =>
						Sreg0 <= RecvPacket_CRC_ARP_S52;
					when RecvPacket_ARP_Payload_Op2 =>
						Sreg0 <= RecvPacket_ARP_Payload_SMac1;
						arp_req_mac(47 downto 40) <= data;
					when RecvPacket_Dest_S22 =>
						Sreg0 <= RecvPacket_Dest_S11;
						dest_mac(39 downto 32) <= data;
					when RecvPacket_Dest_S11 =>
						Sreg0 <= RecvPacket_Dest_S12;
						dest_mac(31 downto 24) <= data;
					when RecvPacket_Dest_S12 =>
						Sreg0 <= RecvPacket_Dest_S13;
						dest_mac(23 downto 16) <= data;
					when RecvPacket_Dest_S13 =>
						Sreg0 <= RecvPacket_Dest_S14;
						dest_mac(15 downto 8) <= data;
					when RecvPacket_Dest_S14 =>
						Sreg0 <= RecvPacket_Dest_S15;
						dest_mac(7 downto 0) <= data;
					when RecvPacket_Dest_S15 =>
						Sreg0 <= RecvPacket_Src_S42;
						src_mac(47 downto 40) <= data;
					when RecvPacket_Src_S42 =>
						Sreg0 <= RecvPacket_Src_S44;
						src_mac(39 downto 32) <= data;
					when RecvPacket_Src_S44 =>
						Sreg0 <= RecvPacket_Src_S43;
						src_mac(31 downto 24) <= data;
					when RecvPacket_Src_S45 =>
						if data = x"08" then
							Sreg0 <= RecvPacket_Type_S26;
						else
							Sreg0 <= RecvPacket_Type_S48;
						end if;
					when RecvPacket_Src_S46 =>
						Sreg0 <= RecvPacket_Src_S45;
						src_mac(7 downto 0) <= data;
					when RecvPacket_Src_S47 =>
						Sreg0 <= RecvPacket_Src_S46;
						src_mac(15 downto 8) <= data;
					when RecvPacket_Src_S43 =>
						Sreg0 <= RecvPacket_Src_S47;
						src_mac(23 downto 16) <= data;
					when RecvPacket_Type_S29 =>
						if is_arp_sig = '1' then
							Sreg0 <= RecvPacket_ARP_Payload_HType1;
						elsif is_ip_sig = '1' then
							Sreg0 <= RecvPacket_IP_Payload_IP_VersionAndHeader;
						else
							Sreg0 <= Idle;
							crc_chk_rd <= '1';
							-- this allows crc_chk to output error status
						end if;
					when RecvPacket_Type_S26 =>
						if data = x"00" then
							Sreg0 <= RecvPacket_Type_S29;
							is_ip_sig <= '1';
						elsif data = x"06" then
							Sreg0 <= RecvPacket_Type_S49;
							is_arp_sig <= '1';
						else
							Sreg0 <= RecvPacket_Type_S1;
						end if;
					when RecvPacket_Type_S1 =>
						if is_arp_sig = '1' then
							Sreg0 <= RecvPacket_ARP_Payload_HType1;
						elsif is_ip_sig = '1' then
							Sreg0 <= RecvPacket_IP_Payload_IP_VersionAndHeader;
						else
							Sreg0 <= Idle;
							crc_chk_rd <= '1';
							-- this allows crc_chk to output error status
						end if;
					when RecvPacket_Type_S48 =>
						Sreg0 <= RecvPacket_Type_S1;
					when RecvPacket_Type_S49 =>
						if is_arp_sig = '1' then
							Sreg0 <= RecvPacket_ARP_Payload_HType1;
						elsif is_ip_sig = '1' then
							Sreg0 <= RecvPacket_IP_Payload_IP_VersionAndHeader;
						else
							Sreg0 <= Idle;
							crc_chk_rd <= '1';
							-- this allows crc_chk to output error status
						end if;
					when RecvPacket_Preamble_S50 =>
						udp_countdown <= x"0007";
						-- idle during preamble reception
						crc_chk_init <= '0';
						Sreg0 <= RecvPacket_Preamble_S57;
					when RecvPacket_Preamble_S54 =>
						Sreg0 <= RecvPacket_Dest_S22;
						four_bit_mode_out <= four_bit_mode;
						-- "permanently" latch four bit mode for outside world
						-- can indicate if 100Mbps vs 1Gbps
						-- but RGMII is handled upstream DIG_GEC.. so appears to be 8_bit mode (after first packet glitch)
						dest_mac(47 downto 40) <= data;
					when RecvPacket_Preamble_S57 =>
						udp_countdown <= udp_countdown -1;
						if udp_countdown = 2 or data = x"D5" then
							Sreg0 <= RecvPacket_Preamble_S54;
							crc_chk_en_unmasked <= '1';
						end if;
					when RecvPacket_CRC_ARP_S52 =>
						udp_countdown <= x"0011";
						Sreg0 <= RecvPacket_CRC_ARP_S53;
					when RecvPacket_CRC_ARP_S53 =>
						udp_countdown <= udp_countdown - 1;
						if udp_countdown = x"0001" then
							Sreg0 <= RecvPacket_CRC_ARP_crc1;
						end if;
					when RecvPacket_CRC_ARP_crc1 =>
						Sreg0 <= RecvPacket_CRC_ARP_crc2;
					when RecvPacket_CRC_ARP_crc2 =>
						Sreg0 <= RecvPacket_CRC_ARP_crc3;
					when RecvPacket_CRC_ARP_crc3 =>
						if dv = '0' then	--packets may be padded before CRC
							Sreg0 <= RecvPacket_CRC_ARP_crc4;
							crc_chk_en_unmasked <= '0';
						end if;
					when RecvPacket_CRC_ARP_crc4 =>
						Sreg0 <= Idle;
						crc_chk_rd <= '1';
						-- this allows crc_chk to output error status
					when RecvPacket_CRC_IP_S55 =>
						udp_countdown(10 downto 0) <= udp_zeros;
						udp_countdown(15 downto 11) <= '0' & x"0";
						if udp_zeros = ("000" & x"00") then
							Sreg0 <= RecvPacket_CRC_IP_crc9;
						else
							Sreg0 <= RecvPacket_CRC_IP_S56;
						end if;
					when RecvPacket_CRC_IP_crc6 =>
						Sreg0 <= Idle;
						crc_chk_rd <= '1';
						-- this allows crc_chk to output error status
					when RecvPacket_CRC_IP_crc7 =>
						if dv = '0' then	--packet might be padded depending on source... so CRC may come later than FSM accounts for
							Sreg0 <= RecvPacket_CRC_IP_crc6;
							crc_chk_en_unmasked <= '0';
						end if;
					when RecvPacket_CRC_IP_crc8 =>
						Sreg0 <= RecvPacket_CRC_IP_crc7;
					when RecvPacket_CRC_IP_crc9 =>
						Sreg0 <= RecvPacket_CRC_IP_crc8;
					when RecvPacket_CRC_IP_S56 =>
						udp_countdown <= udp_countdown - 1;
						if udp_countdown <= x"0001" then	-- '<' added for size 1 case
							Sreg0 <= RecvPacket_CRC_IP_crc9;
						end if;
--vhdl_cover_off
					when others =>
						null;
--vhdl_cover_on
				end case;
			end if;
		end if;
	end if;
end process;

end decipherer_arch;
