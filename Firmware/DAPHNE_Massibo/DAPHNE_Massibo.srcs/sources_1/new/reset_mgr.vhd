----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date:    10:59:20 08/16/2012 
-- Design Name: 
-- Module Name:    reset_mgr - Behavioral 
-- Project Name: 
-- Target Devices: 
-- Tool versions: 
-- Description: 
--
-- Dependencies: 
--
-- Revision: 
-- Revision 0.01 - File Created
-- Additional Comments: 
--
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use IEEE.std_logic_misc.ALL;


entity reset_mgr is
    Port ( slow_clk : in  STD_LOGIC;
           reset_start : in  STD_LOGIC;
           reset : out  STD_LOGIC);
end reset_mgr;

architecture Behavioral of reset_mgr is

	signal cnt : unsigned(15 downto 0) := (others => '0');
	signal old_reset_start : std_logic := '1';
begin

	process(slow_clk)
	begin
	
		if rising_edge(slow_clk) then
		
			reset <= '0';			   
			old_reset_start <= reset_start;
			
			if cnt < 2 then -- 100 -- currently reseting
					reset <= '1';
					cnt <= cnt + 1;
			elsif old_reset_start = '0' and reset_start = '1' then
					cnt <= (others => '0');			
					reset <= '1';
			end if;		
		
		end if;
		
	
	end process;

end Behavioral;

