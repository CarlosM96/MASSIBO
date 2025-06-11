-------------------------------------------------------------------------------
--
-- Title       : Create Packet
-- Design      : ethernet_controller
-- Author      : Ryan Rivera
-- Company     : FNAL
--
-------------------------------------------------------------------------------
--
-- File        : d:\Projects\otsdaq\OtS Ethernet MAC firmware\ActiveHDL_proj\ethernet_controller\compile\create_packet.vhd
-- Generated   : 02/09/16 15:24:07
-- From        : d:/Projects/otsdaq/OtS Ethernet MAC firmware/ActiveHDL_proj/ethernet_controller/src/create_packet.asf
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

use work.params_package.all;
entity create_packet is 
	port (
		addrs: in STD_LOGIC_VECTOR (31 downto 0);
		arp_busy: in STD_LOGIC;
		checksum: in STD_LOGIC_VECTOR (15 downto 0);
		clk: in STD_LOGIC;
		data_length: in STD_LOGIC_VECTOR (10 downto 0);
		dest_ip: in STD_LOGIC_VECTOR (31 downto 0);
		dest_mac: in STD_LOGIC_VECTOR (47 downto 0);
		dest_port: in STD_LOGIC_VECTOR (15 downto 0);
		four_bit_mode: in STD_LOGIC;
		icmp_checksum: in STD_LOGIC_VECTOR (15 downto 0);
		icmp_data: in STD_LOGIC_VECTOR (7 downto 0);
		icmp_ip: in STD_LOGIC_VECTOR (31 downto 0);
		icmp_mac: in STD_LOGIC_VECTOR (47 downto 0);
		icmp_ping: in STD_LOGIC;
		mac: in STD_LOGIC_VECTOR (47 downto 0);
		ping: in STD_LOGIC;
		reset: in STD_LOGIC;
		trigger: in STD_LOGIC;
		busy: out STD_LOGIC;
		checksum_trig: out STD_LOGIC;
		clken_out: out STD_LOGIC;
		crc_gen_en: out STD_LOGIC;
		crc_gen_init: out STD_LOGIC;
		crc_gen_rd: out STD_LOGIC;
		dataout: out STD_LOGIC_VECTOR (7 downto 0);
		en_tx_data: out STD_LOGIC;
		length_count_out: out STD_LOGIC_VECTOR (10 downto 0);
		tx_en: out STD_LOGIC;
		tx_er: out STD_LOGIC;
		tx_icmp_packet: out STD_LOGIC;
		udp_data_sel: out STD_LOGIC);
end create_packet;

architecture create_packet_arch of create_packet is

-- diagram signals declarations
signal clken: STD_LOGIC;
signal delay_count: INTEGER range 0 to 65535;
signal icmp_ping_packet: STD_LOGIC;
signal IP_length: STD_LOGIC_VECTOR (15 downto 0);
signal length_count: STD_LOGIC_VECTOR (10 downto 0);
signal ping_packet: STD_LOGIC;
signal sleep_count: INTEGER range 0 to 65535;
signal test_data: STD_LOGIC_VECTOR (7 downto 0);
signal trigger_sig: STD_LOGIC;
signal UDP_length: STD_LOGIC_VECTOR (15 downto 0);
signal zero_fill_count: STD_LOGIC_VECTOR (10 downto 0);

-- SYMBOLIC ENCODED state machine: Sreg0
type Sreg0_type is (
    SendPacket_Dest_S11, SendPacket_Dest_S12, SendPacket_Dest_S13, SendPacket_Dest_S14, SendPacket_Dest_S15, SendPacket_Src_S16, SendPacket_Src_S18,
    SendPacket_Payload_IP_icmpProtocol, SendPacket_Src_S19, SendPacket_Src_S20, SendPacket_Src_S21, SendPacket_Src_S17, SendPacket_Type_S26,
    SendPacket_Type_S29, SendPacket_Dest_S22, SendPacket_Payload_IP_TotLength2, SendPacket_Payload_IP_VersionAndHeader, SendPacket_Payload_IP_TotLength1,
    SendPacket_Payload_IP_ToS1, SendPacket_Payload_UDP_SourcePort1, SendPacket_Payload_UDP_DestPort1, SendPacket_Payload_IP_ID2, SendPacket_Payload_IP_FlagsAndFrag,
    SendPacket_Payload_IP_FragmentOffset, SendPacket_Payload_IP_icmpTotLength1, SendPacket_Payload_IP_TTL, SendPacket_Payload_IP_Protocol,
    SendPacket_Payload_IP_Checksum1, SendPacket_Payload_IP_Checksum2, SendPacket_Payload_IP_SourceAddr1, SendPacket_Payload_IP_icmpTotLength2,
    SendPacket_Payload_IP_SourceAddr2, SendPacket_Payload_IP_SourceAddr3, SendPacket_Payload_IP_ID1, SendPacket_Payload_IP_SourceAddr4,
    SendPacket_Payload_IP_DestAddr1, SendPacket_Payload_IP_DestAddr2, SendPacket_Payload_IP_DestAddr3, SendPacket_Payload_IP_DestAddr4,
    SendPacket_Payload_UDP_Length1, SendPacket_Payload_UDP_DestPort2, SendPacket_Payload_UDP_SourcePort2, SendPacket_Payload_UDP_Length2,
    SendPacket_Payload_UDP_Checksum1, SendPacket_Payload_UDP_Checksum2, SendPacket_Payload_UDP_DataLoop, Idle, sleep, CheckBusy, SendPacket_Preamble_S58,
    SendPacket_Preamble_S57, SendPacket_CRC_crc4, SendPacket_CRC_crc3, SendPacket_CRC_crc2, SendPacket_CRC_S59, SendPacket_CRC_crc1,
    SendPacket_CRC_S1, SendPacket_Payload_ICMP_Type, SendPacket_Payload_ICMP_Code, SendPacket_Payload_ICMP_Checksum1, SendPacket_Payload_ICMP_ID1,
    SendPacket_Payload_ICMP_Checksum2, SendPacket_Payload_ICMP_ID2, SendPacket_Payload_ICMP_SeqNum1, SendPacket_Payload_ICMP_SeqNum2,
    SendPacket_Payload_ICMP_DataLoop, SendPacket_Dest_S60, SendPacket_Dest_S61, SendPacket_Dest_S62, SendPacket_Dest_S63, SendPacket_Dest_S64,
    SendPacket_Dest_S65, SendPacket_Payload_IP_DestAddr6, SendPacket_Payload_IP_DestAddr5, SendPacket_Payload_IP_DestAddr8, SendPacket_Payload_IP_DestAddr7
);
-- attribute ENUM_ENCODING of Sreg0_type: type is ... -- enum_encoding attribute is not supported for symbolic encoding

signal Sreg0: Sreg0_type;

begin

-- concurrent signals assignments

-- Diagram ACTION
tx_icmp_packet <= icmp_ping_packet;
clken_out <= clken;
-- NOTE: expect trigger is a single clock width pulse (but it is not from Data Manager and seems to be OK)
trigger_sig <= trigger;
four_bit_proc : process (clk) -- make trigger sig a single clock width pulse
begin
	if rising_edge(clk) then
		if (four_bit_mode = '1') then
			clken <= not clken;
		else
			clken <= '1';
		end if;
	end if;
end process;
zero_proc : process(clk)
begin
	if rising_edge(clk) and clken = '1' then
		if trigger_sig = '1' then
			IP_length(15 downto 11) <= (others => '0');
			IP_length(10 downto 0) <= data_length + x"1C";
-- IP length = actual data + 20 (IP header) + 8 (UDP header)
			UDP_length(15 downto 11) <= (others => '0');
			UDP_length(10 downto 0) <= data_length + x"08";
-- UDP length = actual data + 8 (UDP header)
			if data_length < ("000"&x"12") then
				zero_fill_count <= ("000"&x"12") - data_length;
			else
				zero_fill_count <= (others => '0');
			end if;
		elsif icmp_ping_packet = '1' then
-- note: for ICMP ping data_length input = ICMP length
-- ICMP length = actual data + 20 (IP header) + 8(ICMP header)
			if data_length < 46 then
				zero_fill_count <= 46 - data_length;
			else
				zero_fill_count <= (others => '0');
			end if;
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
			checksum_trig <= '0';
			dataout <= x"00";
			udp_data_sel <= '0';
			-- used as select line for output mux and to
-- lock out arp responses
			sleep_count <= 50;
			-- for testing so not triggered twice by accident
			en_tx_data <= '0';
			-- a '1' indicates to the user to provide data on
-- user_tx_data_in(7:0) starting on the next rising edge and
-- continuing on every clock until data is exhausted
			busy <= '0';
			-- if set, indicates that trigger will be ignored
			tx_er <= '0';
			tx_en <= '0';
			crc_gen_en <= '0';
			crc_gen_init <= '0';
			crc_gen_rd <= '0';
			ping_packet <= '0';
			icmp_ping_packet <= '0';
		else
			if clken = '1' then
				-- Set default values for outputs, signals and variables
				checksum_trig <= '0';
				case Sreg0 is
					when Idle =>
						dataout <= x"00";
						udp_data_sel <= '0';
						-- used as select line for output mux and to
						-- lock out arp responses
						sleep_count <= 50;
						-- for testing so not triggered twice by accident
						en_tx_data <= '0';
						-- a '1' indicates to the user to provide data on
						-- user_tx_data_in(7:0) starting on the next rising edge and
						-- continuing on every clock until data is exhausted
						busy <= '0';
						-- if set, indicates that trigger will be ignored
						tx_er <= '0';
						tx_en <= '0';
						crc_gen_en <= '0';
						crc_gen_init <= '0';
						crc_gen_rd <= '0';
						ping_packet <= '0';
						icmp_ping_packet <= '0';
						if trigger_sig = '1' then
							Sreg0 <= CheckBusy;
							length_count <= data_length;
							busy <= '1';
							if ping = '1' then
								ping_packet <= '1';
							end if;
							if icmp_ping = '1' then
								icmp_ping_packet <= '1';
							end if;
						end if;
					when sleep =>
						sleep_count <= sleep_count - 1;
						udp_data_sel <= '0';
						if sleep_count = 1 then
							Sreg0 <= Idle;
						end if;
					when CheckBusy =>
						if arp_busy = '0' then
							Sreg0 <= SendPacket_Preamble_S57;
							udp_data_sel <= '1';
							--sel mux output
							crc_gen_init <= '1';
							-- init crc
							delay_count <= 7;
						end if;
					when SendPacket_Dest_S11 =>
						dataout <= dest_mac(39 downto 32);
						Sreg0 <= SendPacket_Dest_S12;
					when SendPacket_Dest_S12 =>
						dataout <= dest_mac(31 downto 24);
						Sreg0 <= SendPacket_Dest_S13;
					when SendPacket_Dest_S13 =>
						dataout <= dest_mac(23 downto 16);
						Sreg0 <= SendPacket_Dest_S14;
					when SendPacket_Dest_S14 =>
						dataout <= dest_mac(15 downto 8);
						Sreg0 <= SendPacket_Dest_S15;
					when SendPacket_Dest_S15 =>
						dataout <= dest_mac(7 downto 0);
						Sreg0 <= SendPacket_Src_S16;
					when SendPacket_Dest_S22 =>
						dataout <= dest_mac(47 downto 40);
						--first byte of dest MAC
						crc_gen_en <= '1';
						Sreg0 <= SendPacket_Dest_S11;
					when SendPacket_Dest_S60 =>
						dataout <= icmp_mac(39 downto 32);
						Sreg0 <= SendPacket_Dest_S61;
					when SendPacket_Dest_S61 =>
						dataout <= icmp_mac(31 downto 24);
						Sreg0 <= SendPacket_Dest_S62;
					when SendPacket_Dest_S62 =>
						dataout <= icmp_mac(23 downto 16);
						Sreg0 <= SendPacket_Dest_S63;
					when SendPacket_Dest_S63 =>
						dataout <= icmp_mac(15 downto 8);
						Sreg0 <= SendPacket_Dest_S64;
					when SendPacket_Dest_S64 =>
						dataout <= icmp_mac(7 downto 0);
						Sreg0 <= SendPacket_Src_S16;
					when SendPacket_Dest_S65 =>
						dataout <= icmp_mac(47 downto 40);
						--first byte of dest MAC
						crc_gen_en <= '1';
						Sreg0 <= SendPacket_Dest_S60;
					when SendPacket_Src_S16 =>
						dataout <= mac(47 downto 40);
						--x"00";
						Sreg0 <= SendPacket_Src_S18;
					when SendPacket_Src_S18 =>
						dataout <= mac(39 downto 32);
						--x"80";
						Sreg0 <= SendPacket_Src_S17;
					when SendPacket_Src_S19 =>
						dataout <= mac(7 downto 0);
						--addrs;
						Sreg0 <= SendPacket_Type_S26;
					when SendPacket_Src_S20 =>
						dataout <= mac(15 downto 8);
						--x"00";
						Sreg0 <= SendPacket_Src_S19;
					when SendPacket_Src_S21 =>
						dataout <= mac(23 downto 16);
						--x"EC";
						Sreg0 <= SendPacket_Src_S20;
					when SendPacket_Src_S17 =>
						dataout <= mac(31 downto 24);
						--x"55";
						Sreg0 <= SendPacket_Src_S21;
					when SendPacket_Type_S26 =>
						dataout <= x"08";
						Sreg0 <= SendPacket_Type_S29;
					when SendPacket_Type_S29 =>
						dataout <= x"00";
						Sreg0 <= SendPacket_Payload_IP_VersionAndHeader;
					when SendPacket_Payload_UDP_SourcePort1 =>
						dataout <= x"07";
						-- source port is claimed as 2001... but will receive messages to any port
						Sreg0 <= SendPacket_Payload_UDP_SourcePort2;
					when SendPacket_Payload_UDP_DestPort1 =>
						dataout <= dest_port(15 downto 8);
						Sreg0 <= SendPacket_Payload_UDP_DestPort2;
					when SendPacket_Payload_UDP_Length1 =>
						dataout <= UDP_length(15 downto 8);
						Sreg0 <= SendPacket_Payload_UDP_Length2;
					when SendPacket_Payload_UDP_DestPort2 =>
						dataout <= dest_port(7 downto 0);
						Sreg0 <= SendPacket_Payload_UDP_Length1;
					when SendPacket_Payload_UDP_SourcePort2 =>
						dataout <= x"D1";
						Sreg0 <= SendPacket_Payload_UDP_DestPort1;
					when SendPacket_Payload_UDP_Length2 =>
						dataout <= UDP_length(7 downto 0);
						-- 13 bytes = 8 header + 5 data
						Sreg0 <= SendPacket_Payload_UDP_Checksum1;
					when SendPacket_Payload_UDP_Checksum1 =>
						dataout <= x"00";
						-- 0 indicates unused
						Sreg0 <= SendPacket_Payload_UDP_Checksum2;
					when SendPacket_Payload_UDP_Checksum2 =>
						dataout <= x"00";
						test_data <= ETH_CONTROLLER_VERSION(15 downto 8);
						-- x"41";
						-- A
						if ping_packet = '0' then
							en_tx_data <= '1';
							-- indicates to user to have data present on next rising clock edge
						end if;
						Sreg0 <= SendPacket_Payload_UDP_DataLoop;
					when SendPacket_Payload_UDP_DataLoop =>
						length_count <= length_count - 1;
						if test_data = x"5A" then	 -- if Z
						 	test_data <= x"41";
						 	-- A
						else
							test_data <= ETH_CONTROLLER_VERSION(7 downto 0);
							----test_data + 1;
						end if;
						dataout <= test_data;
						if length_count = "000" & x"01" then
							Sreg0 <= SendPacket_CRC_S1;
							en_tx_data <= '0';
							-- a delayed by one clock version of this signal will control the udp data mux
						end if;
					when SendPacket_Payload_IP_icmpProtocol =>
						dataout <= x"01";
						--ICMP
						Sreg0 <= SendPacket_Payload_IP_Checksum1;
					when SendPacket_Payload_IP_TotLength2 =>
						dataout <= IP_length(7 downto 0);
						-- is length in hex of headers and data
						Sreg0 <= SendPacket_Payload_IP_ID1;
					when SendPacket_Payload_IP_VersionAndHeader =>
						dataout <= x"45";
						Sreg0 <= SendPacket_Payload_IP_ToS1;
					when SendPacket_Payload_IP_TotLength1 =>
						dataout <= IP_length(15 downto 8);
						Sreg0 <= SendPacket_Payload_IP_TotLength2;
					when SendPacket_Payload_IP_ToS1 =>
						dataout <= x"00";
						if icmp_ping_packet = '1' then
							Sreg0 <= SendPacket_Payload_IP_icmpTotLength1;
						else
							Sreg0 <= SendPacket_Payload_IP_TotLength1;
						end if;
					when SendPacket_Payload_IP_ID2 =>
						dataout <= x"79";
						Sreg0 <= SendPacket_Payload_IP_FlagsAndFrag;
					when SendPacket_Payload_IP_FlagsAndFrag =>
						dataout <= x"00";
						Sreg0 <= SendPacket_Payload_IP_FragmentOffset;
					when SendPacket_Payload_IP_FragmentOffset =>
						dataout <= x"00";
						Sreg0 <= SendPacket_Payload_IP_TTL;
					when SendPacket_Payload_IP_icmpTotLength1 =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_IP_icmpTotLength2;
					when SendPacket_Payload_IP_TTL =>
						dataout <= x"80";
						-- number of hops allowed
						if icmp_ping_packet = '1' then
							Sreg0 <= SendPacket_Payload_IP_icmpProtocol;
						else
							Sreg0 <= SendPacket_Payload_IP_Protocol;
						end if;
					when SendPacket_Payload_IP_Protocol =>
						dataout <= x"11";
						--UDP
						Sreg0 <= SendPacket_Payload_IP_Checksum1;
					when SendPacket_Payload_IP_Checksum1 =>
						dataout <= checksum(15 downto 8);
						Sreg0 <= SendPacket_Payload_IP_Checksum2;
					when SendPacket_Payload_IP_Checksum2 =>
						dataout <= checksum(7 downto 0);
						Sreg0 <= SendPacket_Payload_IP_SourceAddr1;
					when SendPacket_Payload_IP_SourceAddr1 =>
						dataout <= addrs(31 downto 24);
						--x"C0";
						Sreg0 <= SendPacket_Payload_IP_SourceAddr2;
					when SendPacket_Payload_IP_icmpTotLength2 =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_IP_ID1;
					when SendPacket_Payload_IP_SourceAddr2 =>
						dataout <= addrs(23 downto 16);
						--x"A8";
						Sreg0 <= SendPacket_Payload_IP_SourceAddr3;
					when SendPacket_Payload_IP_SourceAddr3 =>
						dataout <= addrs(15 downto 8);
						--x"85";
						Sreg0 <= SendPacket_Payload_IP_SourceAddr4;
					when SendPacket_Payload_IP_ID1 =>
						dataout <= x"35";
						Sreg0 <= SendPacket_Payload_IP_ID2;
					when SendPacket_Payload_IP_SourceAddr4 =>
						dataout <= addrs(7 downto 0);
						--addrs;
						if icmp_ping_packet = '1' then
							Sreg0 <= SendPacket_Payload_IP_DestAddr6;
						else
							Sreg0 <= SendPacket_Payload_IP_DestAddr1;
						end if;
					when SendPacket_Payload_IP_DestAddr1 =>
						dataout <= dest_ip(31 downto 24);
						-- 192.168.133.1 => 0 x C0 A8 85 01
						Sreg0 <= SendPacket_Payload_IP_DestAddr2;
					when SendPacket_Payload_IP_DestAddr2 =>
						dataout <= dest_ip(23 downto 16);
						Sreg0 <= SendPacket_Payload_IP_DestAddr3;
					when SendPacket_Payload_IP_DestAddr3 =>
						dataout <= dest_ip(15 downto 8);
						Sreg0 <= SendPacket_Payload_IP_DestAddr4;
					when SendPacket_Payload_IP_DestAddr4 =>
						dataout <= dest_ip(7 downto 0);
						if icmp_ping_packet = '1' then
							Sreg0 <= SendPacket_Payload_ICMP_Type;
						else
							Sreg0 <= SendPacket_Payload_UDP_SourcePort1;
						end if;
					when SendPacket_Payload_IP_DestAddr6 =>
						dataout <= icmp_ip(31 downto 24);
						-- 192.168.133.1 => 0 x C0 A8 85 01
						Sreg0 <= SendPacket_Payload_IP_DestAddr5;
					when SendPacket_Payload_IP_DestAddr5 =>
						dataout <= icmp_ip(23 downto 16);
						Sreg0 <= SendPacket_Payload_IP_DestAddr8;
					when SendPacket_Payload_IP_DestAddr8 =>
						dataout <= icmp_ip(15 downto 8);
						Sreg0 <= SendPacket_Payload_IP_DestAddr7;
					when SendPacket_Payload_IP_DestAddr7 =>
						dataout <= icmp_ip(7 downto 0);
						if icmp_ping_packet = '1' then
							Sreg0 <= SendPacket_Payload_ICMP_Type;
						else
							Sreg0 <= SendPacket_Payload_UDP_SourcePort1;
						end if;
					when SendPacket_Payload_ICMP_Type =>
						dataout <= x"00";
						-- Type: ICMP Ping Response (0x08 is Request)
						length_count <= data_length - 20;
						--get icmp payload count
						Sreg0 <= SendPacket_Payload_ICMP_Code;
					when SendPacket_Payload_ICMP_Code =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_ICMP_Checksum1;
					when SendPacket_Payload_ICMP_Checksum1 =>
						dataout <= icmp_checksum(15 downto 8);
						Sreg0 <= SendPacket_Payload_ICMP_Checksum2;
					when SendPacket_Payload_ICMP_ID1 =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_ICMP_ID2;
					when SendPacket_Payload_ICMP_Checksum2 =>
						dataout <= icmp_checksum(7 downto 0);
						Sreg0 <= SendPacket_Payload_ICMP_ID1;
					when SendPacket_Payload_ICMP_ID2 =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_ICMP_SeqNum1;
					when SendPacket_Payload_ICMP_SeqNum1 =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_ICMP_SeqNum2;
					when SendPacket_Payload_ICMP_SeqNum2 =>
						dataout <= icmp_data;
						Sreg0 <= SendPacket_Payload_ICMP_DataLoop;
					when SendPacket_Payload_ICMP_DataLoop =>
						length_count <= length_count - 1;
						dataout <= icmp_data;
						if length_count = "000" & x"01" then
							Sreg0 <= SendPacket_CRC_S1;
							en_tx_data <= '0';
							-- a delayed by one clock version of this signal will control the udp data mux
						end if;
					when SendPacket_Preamble_S58 =>
						dataout <= x"D5";
						if icmp_ping_packet = '1' then
							Sreg0 <= SendPacket_Dest_S65;
							checksum_trig <= '1';
							--latch length for checksum
							length_count_out <= data_length-20;
							--the length comes a little late.. since the icmp ping is trying to be as responsive as possible
						else
							Sreg0 <= SendPacket_Dest_S22;
							checksum_trig <= '1';
							--latch length for checksum
							length_count_out <= length_count;
							--latch the already latched value for checksum
						end if;
					when SendPacket_Preamble_S57 =>
						delay_count <= delay_count - 1;
						dataout <= x"55";
						tx_en <= '1';
						crc_gen_init <= '0';
						if delay_count = 1 then
							Sreg0 <= SendPacket_Preamble_S58;
						end if;
					when SendPacket_CRC_crc4 =>
						Sreg0 <= sleep;
						crc_gen_rd <= '0';
						tx_en <= '0';
					when SendPacket_CRC_crc3 =>
						Sreg0 <= SendPacket_CRC_crc4;
					when SendPacket_CRC_crc2 =>
						Sreg0 <= SendPacket_CRC_crc3;
					when SendPacket_CRC_S59 =>
						delay_count <= delay_count - 1;
						if delay_count = 1 then
							Sreg0 <= SendPacket_CRC_crc1;
							crc_gen_rd <= '1';
							crc_gen_en <= '0';
						end if;
					when SendPacket_CRC_crc1 =>
						Sreg0 <= SendPacket_CRC_crc2;
					when SendPacket_CRC_S1 =>
						dataout <= x"00";
						if zero_fill_count = ("000" & x"00") then
							Sreg0 <= SendPacket_CRC_crc1;
							crc_gen_rd <= '1';
							crc_gen_en <= '0';
						else
							Sreg0 <= SendPacket_CRC_S59;
							delay_count <= conv_integer(zero_fill_count);
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

end create_packet_arch;
