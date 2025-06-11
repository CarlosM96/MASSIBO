-------------------------------------------------------------------------------
--
-- Title       : Burst Controller State Machine
-- Design      : burst_controller
-- Author      : Ryan Rivera
-- Company     : FNAL
--
--------------------------------
--
-- Description : 
--
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.std_logic_arith.all;
use IEEE.std_logic_unsigned.all;

entity burst_controller_sm is 
	port (					  
		clk: in STD_LOGIC;
		reset: in STD_LOGIC;
									 					  
		b_mode: in STD_LOGIC; 
		
		b_data_we: in STD_LOGIC;	 
		b_end_packet: in STD_LOGIC;	  
		tx_data_full: in STD_LOGIC;
		tx_info_full: in STD_LOGIC;	   
		
		b_enable: out STD_LOGIC;   
		
		tx_data_we: out STD_LOGIC;
		tx_info: out STD_LOGIC_VECTOR (15 downto 0);
		tx_info_we: out STD_LOGIC);
end burst_controller_sm;

architecture burst_controller_sm_arch of burst_controller_sm is

	constant max_packet_64_size: STD_LOGIC_VECTOR (7 downto 0) := x"B6";  -- 182;	

	-- diagram signals declarations
	signal b_enable_sig: STD_LOGIC;
	signal b_packet_qw_size: STD_LOGIC_VECTOR (7 downto 0);	   --this count should always be 
						-- equal to data_manager/burst_traffic_controller/writes_in_curr_burst
	signal first_packet_sig, b_end_packet_old: STD_LOGIC;
	signal just_reset: STD_LOGIC;
	signal reset_packet_size: STD_LOGIC;
	
	-- SYMBOLIC ENCODED state machine: Sreg0
	type Sreg0_type is (
	    Wait_for_End, Reset_Size, Idle
	);
	-- attribute enum_encoding of Sreg0_type: type is ... -- enum_encoding attribute is not supported for symbolic encoding	  	
	signal Sreg0: Sreg0_type;

begin
	
	-- concurrent signals assignments		 
	
	-- this block only affects first 2 bits of packet type:  1 = first in burst, 2 = middle, 3 = last in burst
	tx_info(7 downto 2) <= (others => '0');
	
	-- don't allow writes when tx fifo is full (when overflow data is dropped on the floor)
	tx_data_we <= b_enable_sig and b_data_we and (not tx_data_full) and (not tx_info_full);	
	b_enable <= b_enable_sig and (not tx_data_full) and (not tx_info_full);						  
	
								  
	----------------------------------------------------------------------
	-- Proc_size: 		 
	--  counts the number of quad-words
	-- in the current packet and saves the value
	-- in b_packet_qw_size.
	----------------------------------------------------------------------	 	
	proc_size: process(clk)
	begin										   
		if rising_edge(clk) then			
		
			if reset = '1' then
				b_packet_qw_size <= (others => '0');
			else
				just_reset <= '0';
				if b_enable_sig = '1' and b_data_we = '1' and tx_data_full = '0' and tx_info_full = '0' then
					b_packet_qw_size <= b_packet_qw_size + 1;
					if b_packet_qw_size = max_packet_64_size or b_end_packet = '1' then -- include case where packet is ended at same time as b_we pulse
						b_packet_qw_size <= x"01";
						-- start next packet (allows for burst to not use b_end_packet signal)
						just_reset <= '1';
					end if;
				elsif b_enable_sig = '1' and b_end_packet = '1' then
					b_packet_qw_size <= (others => '0');
				elsif just_reset = '0' and reset_packet_size = '1' then	  
					-- follow reset_packet_size flag, if didn't just reset
					b_packet_qw_size <= (others => '0');					  
				end if;		 
			end if;
			
		end if;
	end process;
	
	----------------------------------------------------------------------
	-- Machine: Sreg0
	----------------------------------------------------------------------
	Sreg0_machine: process (clk)
	begin
		if rising_edge(clk) then	   
			  					
			-- Set default values for outputs, signals and variables
			-- ...
			tx_info_we <= '0';	  
			reset_packet_size <= '0';	  
			b_end_packet_old <= b_end_packet;
			
			if reset = '1' then	
				Sreg0 <= Idle;
				-- Set reset values for outputs, signals and variables
				-- ...
				tx_info(1 downto 0) <= "01";
				b_enable_sig <= '0'; 
				first_packet_sig <= '1';	 
				tx_info(15 downto 8) <= (others => '0');
			else						  
				case Sreg0 is						  
					when Wait_for_End =>
						if b_mode = '0' then 	  
							Sreg0 <= Reset_Size;
							reset_packet_size <= '1';
							tx_info(15 downto 8) <= b_packet_qw_size;
							tx_info(1 downto 0) <= "11";  --indicate last in burst
							tx_info_we <= '1';
							b_enable_sig <= '0';
						elsif ( (b_end_packet_old = '0' and b_end_packet = '1') or 
							b_packet_qw_size = max_packet_64_size ) then	-- end of packet detected	 						
							Sreg0 <= Wait_for_End;
							tx_info(15 downto 8) <= b_packet_qw_size;	
							
							if first_packet_sig = '1' then
								tx_info(1 downto 0) <= "01";  --indicate first in burst
								first_packet_sig <= '0';   
							else		 
								tx_info(1 downto 0) <= "10";  --indicate middle of burst								
							end if;			   			
							
							tx_info_we <= '1';
							reset_packet_size <= '1';
						end if;			
					when Reset_Size =>
						Sreg0 <= Idle;						   
						b_enable_sig <= '0';	
						first_packet_sig <= '1';	
					when Idle =>
						if b_mode = '1' then	
							Sreg0 <= Wait_for_End;
							b_enable_sig <= '1';   
						end if;		
					when others =>
						null;	 
				end case;
			end if;
		end if;
	end process;

end burst_controller_sm_arch;
