-------------------------------------------------------------------------------
--
-- Title       : No Title
-- Design      : ethernet_controller
-- Author      : rrivera
-- Company     : Fermi National Accelerator Laboratory
--
-------------------------------------------------------------------------------
--
-- File        : d:\Projects\otsdaq\OtS Ethernet MAC firmware\ActiveHDL_proj\ethernet_controller\compile\icmp_ping_checksum_calc.vhd
-- Generated   : 02/08/16 17:09:23
-- From        : d:/Projects/otsdaq/OtS Ethernet MAC firmware/ActiveHDL_proj/ethernet_controller/src/icmp_ping_checksum_calc.asf
-- By          : FSM2VHDL ver. 5.0.7.2
--
-------------------------------------------------------------------------------
--
-- Description : 
--
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use ieee.numeric_std.all;		

entity icmp_ping_checksum_calc is 
	port (
		clk: in STD_LOGIC;
		req_chk_sum: in STD_LOGIC_VECTOR (15 downto 0);
		reset: in STD_LOGIC;
		trigger: in STD_LOGIC;
		resp_chk_sum: out STD_LOGIC_VECTOR (15 downto 0));
end icmp_ping_checksum_calc;

architecture icmp_ping_checksum_calc_arch of icmp_ping_checksum_calc is

-- diagram signals declarations
signal req_chk_sum_sig: UNSIGNED (16 downto 0);

-- SYMBOLIC ENCODED state machine: Sreg0
type Sreg0_type is (
    Init, S1, S2, S3, S4
);
-- attribute ENUM_ENCODING of Sreg0_type: type is ... -- enum_encoding attribute is not supported for symbolic encoding

signal Sreg0: Sreg0_type;

begin

-- concurrent signals assignments

-- Diagram ACTION

----------------------------------------------------------------------
-- Machine: Sreg0
----------------------------------------------------------------------
Sreg0_machine: process (clk)
begin
	if clk'event and clk = '1' then
		if reset = '1' then
			Sreg0 <= Init;
			-- Set default values for outputs, signals and variables
			-- ...
		else
			-- Set default values for outputs, signals and variables
			-- ...
			case Sreg0 is
				when Init =>
					Sreg0 <= S1;
					req_chk_sum_sig <= unsigned('0' & (not req_chk_sum));
				when S1 =>
					Sreg0 <= S2;
					req_chk_sum_sig <= req_chk_sum_sig + ('0' & x"F7FF");
				when S2 =>
					Sreg0 <= S3;
					if req_chk_sum_sig(16) = '1' then
					  	req_chk_sum_sig(15 downto 0) <= req_chk_sum_sig(15 downto 0) + 1;
					  	req_chk_sum_sig(16) <= '0';
					end if;
				when S3 =>
					Sreg0 <= S4;
					resp_chk_sum <= not (std_logic_vector(req_chk_sum_sig(15 downto 0)));
				when S4 =>
					Sreg0 <= Init;
--vhdl_cover_off
				when others =>
					null;
--vhdl_cover_on
			end case;
		end if;
	end if;
end process;

end icmp_ping_checksum_calc_arch;
