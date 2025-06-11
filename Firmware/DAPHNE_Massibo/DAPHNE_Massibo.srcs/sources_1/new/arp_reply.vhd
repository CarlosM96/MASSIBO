-------------------------------------------------------------------------------
--
-- Title       : Arp Reply
-- Design      : ethernet_controller
-- Author      : Ryan Rivera
-- Company     : FNAL
--
-------------------------------------------------------------------------------
--
-- File        : d:\Projects\otsdaq\OtS Ethernet MAC firmware\ActiveHDL_proj\ethernet_controller\compile\arp_reply.vhd
-- Generated   : 02/08/16 17:09:25
-- From        : d:/Projects/otsdaq/OtS Ethernet MAC firmware/ActiveHDL_proj/ethernet_controller/src/arp_reply.asf
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

entity arp_reply is 
	port (
		addrs: in STD_LOGIC_VECTOR (31 downto 0);
		arp_announce: in STD_LOGIC;
		clk: in STD_LOGIC;
		four_bit_mode: in STD_LOGIC;
		mac: in STD_LOGIC_VECTOR (47 downto 0);
		reset: in STD_LOGIC;
		tip: in STD_LOGIC_VECTOR (31 downto 0);
		tmac: in STD_LOGIC_VECTOR (47 downto 0);
		trigger: in STD_LOGIC;
		udp_busy: in STD_LOGIC;
		arp_busy: out STD_LOGIC;
		crc_gen_en: out STD_LOGIC;
		crc_gen_init: out STD_LOGIC;
		crc_gen_rd: out STD_LOGIC;
		dataout: out STD_LOGIC_VECTOR (7 downto 0);
		tx_en: out STD_LOGIC;
		tx_er: out STD_LOGIC);
end arp_reply;

architecture arp_reply_arch of arp_reply is

-- diagram signals declarations
signal announce_sig: STD_LOGIC;
signal clken: STD_LOGIC;
signal delay_count: INTEGER range 0 to 65535;
signal old_trig: STD_LOGIC;
signal trigger_sig: STD_LOGIC;

-- SYMBOLIC ENCODED state machine: Sreg0
type Sreg0_type is (
    idle, delay, ArpResponse_ARP_Payload_PType2, ArpResponse_ARP_Payload_HLen, ArpResponse_ARP_Payload_PLen, ArpResponse_ARP_Payload_SMac1,
    ArpResponse_ARP_Payload_SMac2, ArpResponse_ARP_Payload_SMac3, ArpResponse_ARP_Payload_SMac4, ArpResponse_ARP_Payload_SMac5, ArpResponse_ARP_Payload_SMac6,
    ArpResponse_ARP_Payload_Sip1, ArpResponse_ARP_Payload_Sip2, ArpResponse_ARP_Payload_Sip3, ArpResponse_ARP_Payload_Sip4, ArpResponse_ARP_Payload_TMac1,
    ArpResponse_ARP_Payload_TMac2, ArpResponse_ARP_Payload_TMac3, ArpResponse_ARP_Payload_TMac4, ArpResponse_ARP_Payload_TMac5, ArpResponse_ARP_Payload_TMac6,
    ArpResponse_ARP_Payload_Tip1, ArpResponse_ARP_Payload_Tip2, ArpResponse_ARP_Payload_Tip3, ArpResponse_Type_S1, ArpResponse_Type_S48,
    ArpResponse_Dest_S22, ArpResponse_Dest_S11, ArpResponse_Dest_S12, ArpResponse_Dest_S13, ArpResponse_Dest_S14, ArpResponse_Dest_S15,
    ArpResponse_ARP_Payload_HType1, ArpResponse_ARP_Payload_Tip4, ArpResponse_ARP_Payload_HType2, ArpResponse_ARP_Payload_PType1, ArpResponse_Src_S50,
    ArpResponse_Src_S51, ArpResponse_Src_S52, ArpResponse_Src_S53, ArpResponse_Src_S54, ArpResponse_ARP_Payload_Op1, ArpResponse_Src_S55,
    ArpResponse_ARP_Payload_Op2, delay1, CheckBusy, ArpResponse_Preamble_S57, ArpResponse_Preamble_S58, ArpResponse_CRC_crc1, ArpResponse_CRC_crc2,
    ArpResponse_CRC_crc3, ArpResponse_CRC_crc4, ArpResponse_CRC_S59, PreCheckBusy, ArpResponse_ARP_Payload_Op3, ArpResponse_ARP_Payload_Op4,
    ArpResponse_ARP_Payload_TMac7, ArpResponse_ARP_Payload_TMac8, ArpResponse_ARP_Payload_TMac9, ArpResponse_ARP_Payload_TMac10, ArpResponse_ARP_Payload_TMac11,
    ArpResponse_ARP_Payload_TMac12, ArpResponse_ARP_Payload_Tip5, ArpResponse_ARP_Payload_Tip6, ArpResponse_ARP_Payload_Tip7, ArpResponse_ARP_Payload_Tip8
);
-- attribute ENUM_ENCODING of Sreg0_type: type is ... -- enum_encoding attribute is not supported for symbolic encoding

signal Sreg0: Sreg0_type;

begin

-- concurrent signals assignments

-- Diagram ACTION
-- Note: expect that trigger is a single clock width pulse
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

----------------------------------------------------------------------
-- Machine: Sreg0
----------------------------------------------------------------------
Sreg0_machine: process (clk)
begin
	if clk'event and clk = '1' then
		if reset = '1' then
			Sreg0 <= idle;
			-- Set default values for outputs, signals and variables
			-- ...
			delay_count <= 20;
			-- number of ticks after trigger to delay sending of packet
-- also used for loops within packet
			arp_busy <= '0';
			-- indicates to udp to not take over transmit line
			dataout <= (others => '0');
			tx_er <= '0';
			tx_en <= '0';
			crc_gen_en <= '0';
			crc_gen_init <= '0';
			crc_gen_rd <= '0';
			announce_sig <= '0';
		else
			if clken = '1' then
				-- Set default values for outputs, signals and variables
				-- ...
				case Sreg0 is
					when idle =>
						delay_count <= 20;
						-- number of ticks after trigger to delay sending of packet
						-- also used for loops within packet
						arp_busy <= '0';
						-- indicates to udp to not take over transmit line
						dataout <= (others => '0');
						tx_er <= '0';
						tx_en <= '0';
						crc_gen_en <= '0';
						crc_gen_init <= '0';
						crc_gen_rd <= '0';
						announce_sig <= '0';
						if trigger_sig = '1' then
							Sreg0 <= PreCheckBusy;
							arp_busy <= '1';
							if (arp_announce = '1') then
							 announce_sig <= '1';
							end if;
						end if;
					when delay =>
						delay_count <= delay_count - 1;
						if delay_count = 1 then
							Sreg0 <= ArpResponse_Preamble_S57;
							crc_gen_init <= '0';
							dataout <= x"55";
							tx_en <= '1';
							delay_count <= 7;
						end if;
					when delay1 =>
						delay_count <= delay_count - 1;
						if delay_count = 1 then
							Sreg0 <= idle;
						end if;
					when CheckBusy =>
						if udp_busy = '0' then
							Sreg0 <= delay;
							crc_gen_init <= '1';
						end if;
					when PreCheckBusy =>
						if udp_busy = '0' then
							Sreg0 <= CheckBusy;
						end if;
					when ArpResponse_Dest_S22 =>
						Sreg0 <= ArpResponse_Dest_S11;
						if(announce_sig = '0') then
							dataout <= tmac(39 downto 32);
						else
							dataout <= x"FF";
						end if;
					when ArpResponse_Dest_S11 =>
						Sreg0 <= ArpResponse_Dest_S12;
						if(announce_sig = '0') then
							dataout <= tmac(31 downto 24);
						else
							dataout <= x"FF";
						end if;
					when ArpResponse_Dest_S12 =>
						Sreg0 <= ArpResponse_Dest_S13;
						if(announce_sig = '0') then
							dataout <= tmac(23 downto 16);
						else
							dataout <= x"FF";
						end if;
					when ArpResponse_Dest_S13 =>
						Sreg0 <= ArpResponse_Dest_S14;
						if(announce_sig = '0') then
							dataout <= tmac(15 downto 8);
						else
							dataout <= x"FF";
						end if;
					when ArpResponse_Dest_S14 =>
						Sreg0 <= ArpResponse_Dest_S15;
						if(announce_sig = '0') then
							dataout <= tmac(7 downto 0);
						else
							dataout <= x"FF";
						end if;
					when ArpResponse_Dest_S15 =>
						Sreg0 <= ArpResponse_Src_S51;
						dataout <= mac(47 downto 40);
						--x"00";
					when ArpResponse_Src_S50 =>
						Sreg0 <= ArpResponse_Src_S55;
						dataout <= mac(31 downto 24);
						--x"55";
					when ArpResponse_Src_S51 =>
						Sreg0 <= ArpResponse_Src_S50;
						dataout <= mac(39 downto 32);
						--x"80";
					when ArpResponse_Src_S52 =>
						Sreg0 <= ArpResponse_Type_S48;
						dataout <= x"08";
					when ArpResponse_Src_S53 =>
						Sreg0 <= ArpResponse_Src_S52;
						dataout <= mac(7 downto 0);
						--addrs;
					when ArpResponse_Src_S54 =>
						Sreg0 <= ArpResponse_Src_S53;
						dataout <= mac(15 downto 8);
						--x"00";
					when ArpResponse_Src_S55 =>
						Sreg0 <= ArpResponse_Src_S54;
						dataout <= mac(23 downto 16);
						--x"EC";
					when ArpResponse_Type_S1 =>
						Sreg0 <= ArpResponse_ARP_Payload_HType1;
						dataout <= x"00";
					when ArpResponse_Type_S48 =>
						Sreg0 <= ArpResponse_Type_S1;
						dataout <= x"06";
					when ArpResponse_ARP_Payload_PType2 =>
						Sreg0 <= ArpResponse_ARP_Payload_HLen;
						dataout <= x"06";
					when ArpResponse_ARP_Payload_HLen =>
						Sreg0 <= ArpResponse_ARP_Payload_PLen;
						dataout <= x"04";
					when ArpResponse_ARP_Payload_PLen =>
						if announce_sig = '1' then
							Sreg0 <= ArpResponse_ARP_Payload_Op3;
							dataout <= x"00";
							-- ARP reply
						else
							Sreg0 <= ArpResponse_ARP_Payload_Op1;
							dataout <= x"00";
							-- ARP reply
						end if;
					when ArpResponse_ARP_Payload_SMac1 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac2;
						dataout <= mac(39 downto 32);
						--x"80";
					when ArpResponse_ARP_Payload_SMac2 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac3;
						dataout <= mac(31 downto 24);
						--x"55";
					when ArpResponse_ARP_Payload_SMac3 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac4;
						dataout <= mac(23 downto 16);
						--x"EC";
					when ArpResponse_ARP_Payload_SMac4 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac5;
						dataout <= mac(15 downto 8);
						--x"00";
					when ArpResponse_ARP_Payload_SMac5 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac6;
						dataout <= mac(7 downto 0);
						--addrs;
					when ArpResponse_ARP_Payload_SMac6 =>
						Sreg0 <= ArpResponse_ARP_Payload_Sip1;
						dataout <= addrs(31 downto 24);
						--x"C0";
					when ArpResponse_ARP_Payload_Sip1 =>
						Sreg0 <= ArpResponse_ARP_Payload_Sip2;
						dataout <= addrs(23 downto 16);
						--x"A8";
					when ArpResponse_ARP_Payload_Sip2 =>
						Sreg0 <= ArpResponse_ARP_Payload_Sip3;
						dataout <= addrs(15 downto 8);
						--x"85";
					when ArpResponse_ARP_Payload_Sip3 =>
						Sreg0 <= ArpResponse_ARP_Payload_Sip4;
						dataout <= addrs(7 downto 0);
						--addrs;
					when ArpResponse_ARP_Payload_Sip4 =>
						if announce_sig = '1' then
							Sreg0 <= ArpResponse_ARP_Payload_TMac7;
							dataout <= x"00";
						else
							Sreg0 <= ArpResponse_ARP_Payload_TMac1;
							dataout <= tmac(47 downto 40);
						end if;
					when ArpResponse_ARP_Payload_TMac1 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac2;
						dataout <= tmac(39 downto 32);
					when ArpResponse_ARP_Payload_TMac2 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac3;
						dataout <= tmac(31 downto 24);
					when ArpResponse_ARP_Payload_TMac3 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac4;
						dataout <= tmac(23 downto 16);
					when ArpResponse_ARP_Payload_TMac4 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac5;
						dataout <= tmac(15 downto 8);
					when ArpResponse_ARP_Payload_TMac5 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac6;
						dataout <= tmac(7 downto 0);
					when ArpResponse_ARP_Payload_TMac6 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip1;
						dataout <= tip(31 downto 24);
					when ArpResponse_ARP_Payload_Tip1 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip2;
						dataout <= tip(23 downto 16);
					when ArpResponse_ARP_Payload_Tip2 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip3;
						dataout <= tip(15 downto 8);
					when ArpResponse_ARP_Payload_Tip3 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip4;
						dataout <= tip(7 downto 0);
					when ArpResponse_ARP_Payload_HType1 =>
						Sreg0 <= ArpResponse_ARP_Payload_HType2;
						dataout <= x"01";
						--ethernet
					when ArpResponse_ARP_Payload_Tip4 =>
						Sreg0 <= ArpResponse_CRC_S59;
						dataout <= (others => '0');
						-- crc may need buffered 0 input
						delay_count <= 18;
					when ArpResponse_ARP_Payload_HType2 =>
						Sreg0 <= ArpResponse_ARP_Payload_PType1;
						dataout <= x"08";
						--IPv4
					when ArpResponse_ARP_Payload_PType1 =>
						Sreg0 <= ArpResponse_ARP_Payload_PType2;
						dataout <= x"00";
					when ArpResponse_ARP_Payload_Op1 =>
						Sreg0 <= ArpResponse_ARP_Payload_Op2;
						dataout <= x"02";
						-- ARP reply
					when ArpResponse_ARP_Payload_Op2 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac1;
						dataout <= mac(47 downto 40);
						--x"00";
					when ArpResponse_ARP_Payload_Op3 =>
						Sreg0 <= ArpResponse_ARP_Payload_Op4;
						dataout <= x"01";
						-- ARP request
					when ArpResponse_ARP_Payload_Op4 =>
						Sreg0 <= ArpResponse_ARP_Payload_SMac1;
						dataout <= mac(47 downto 40);
						--x"00";
					when ArpResponse_ARP_Payload_TMac7 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac8;
						dataout <= x"00";
					when ArpResponse_ARP_Payload_TMac8 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac9;
						dataout <= x"00";
					when ArpResponse_ARP_Payload_TMac9 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac10;
						dataout <= x"00";
					when ArpResponse_ARP_Payload_TMac10 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac11;
						dataout <= x"00";
					when ArpResponse_ARP_Payload_TMac11 =>
						Sreg0 <= ArpResponse_ARP_Payload_TMac12;
						dataout <= x"00";
					when ArpResponse_ARP_Payload_TMac12 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip5;
						dataout <= addrs(31 downto 24);
						--x"C0";
					when ArpResponse_ARP_Payload_Tip5 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip6;
						dataout <= addrs(23 downto 16);
						--x"A8";
					when ArpResponse_ARP_Payload_Tip6 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip7;
						dataout <= addrs(15 downto 8);
						--x"85";
					when ArpResponse_ARP_Payload_Tip7 =>
						Sreg0 <= ArpResponse_ARP_Payload_Tip8;
						dataout <= addrs(7 downto 0);
						--addrs;
					when ArpResponse_ARP_Payload_Tip8 =>
						Sreg0 <= ArpResponse_CRC_S59;
						dataout <= (others => '0');
						-- crc may need buffered 0 input
						delay_count <= 18;
					when ArpResponse_Preamble_S57 =>
						delay_count <= delay_count - 1;
						if delay_count = 1 then
							Sreg0 <= ArpResponse_Preamble_S58;
							dataout <= x"D5";
						end if;
					when ArpResponse_Preamble_S58 =>
						Sreg0 <= ArpResponse_Dest_S22;
						if(announce_sig = '0') then
							dataout <= tmac(47 downto 40);
						else
							dataout <= x"FF";
						end if;
						crc_gen_en <= '1';
					when ArpResponse_CRC_crc1 =>
						Sreg0 <= ArpResponse_CRC_crc2;
					when ArpResponse_CRC_crc2 =>
						Sreg0 <= ArpResponse_CRC_crc3;
					when ArpResponse_CRC_crc3 =>
						Sreg0 <= ArpResponse_CRC_crc4;
					when ArpResponse_CRC_crc4 =>
						Sreg0 <= delay1;
						crc_gen_rd <= '0';
						tx_en <= '0';
						delay_count <= 30;
						-- number of ticks after sending packet to keep line dead from udp
					when ArpResponse_CRC_S59 =>
						delay_count <= delay_count - 1;
						if delay_count = 1 then
							Sreg0 <= ArpResponse_CRC_crc1;
							crc_gen_rd <= '1';
							crc_gen_en <= '0';
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

end arp_reply_arch;
