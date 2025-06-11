-------------------------------------------------------------------------------
--
-- Title       : GEC_RX_CTL
-- Design      : ethernet_controller
-- Author      : aprosser
-- Company     : CD_CEPA_ESE
--
-------------------------------------------------------------------------------
--
-- File        : d:\Projects\otsdaq\OtS Ethernet MAC firmware\ActiveHDL_proj\ethernet_controller\compile\rx_ctl.vhd
-- Generated   : 02/09/16 09:35:31
-- From        : d:/Projects/otsdaq/OtS Ethernet MAC firmware/ActiveHDL_proj/ethernet_controller/src/rx_ctl.asf
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
use work.params_package.all;

entity rx_ctl is 
	port (
		clear_crc_err_flag: in STD_LOGIC;
		clock: in STD_LOGIC;
		four_bit_mode: in STD_LOGIC;
		reset: in STD_LOGIC;
		user_crc_chk: in STD_LOGIC;
		user_crc_err: in STD_LOGIC;
		user_rx_data_out: in STD_LOGIC_VECTOR (7 downto 0);
		user_rx_valid_out: in STD_LOGIC;
		crc_err_flag: out STD_LOGIC;
		data_fifo_wdata: out STD_LOGIC_VECTOR (63 downto 0);
		data_fifo_wren: out STD_LOGIC;
		info_fifo_wr_data: out STD_LOGIC_VECTOR (15 downto 0);
		info_fifo_wren: out STD_LOGIC);
end rx_ctl;

architecture arch of rx_ctl is

-- diagram signals declarations
signal clken: STD_LOGIC;
signal com_code: STD_LOGIC;
signal crc_err_reg: STD_LOGIC;
signal data_fifo_wdata_sig: STD_LOGIC_VECTOR (63 downto 0);
signal data_fifo_wren_sig: STD_LOGIC;
signal info_fifo_wren_sig: STD_LOGIC;
signal q_w_count: UNSIGNED (7 downto 0);
signal q_w_counter: UNSIGNED (7 downto 0);

-- BINARY ENCODED state machine: Sreg0
attribute ENUM_ENCODING: string;
type Sreg0_type is (
    idle, insert_crc, rcvdone, S13, S3_S4, S3_S9, S3_S8, S3_S7, S3_S6, S3_S5, S3_S11, S3_S10
);
attribute ENUM_ENCODING of Sreg0_type: type is
	"0000 " &		-- idle
	"0001 " &		-- insert_crc
	"0010 " &		-- rcvdone
	"0011 " &		-- S13
	"0100 " &		-- S3_S4
	"0101 " &		-- S3_S9
	"0110 " &		-- S3_S8
	"0111 " &		-- S3_S7
	"1000 " &		-- S3_S6
	"1001 " &		-- S3_S5
	"1010 " &		-- S3_S11
	"1011" ;		-- S3_S10

signal Sreg0: Sreg0_type;

attribute STATE_VECTOR: string;
attribute STATE_VECTOR of arch: architecture is "Sreg0";

begin

-- concurrent signals assignments

-- Diagram ACTION
info_fifo_wren <= info_fifo_wren_sig and clken;
data_fifo_wren <= data_fifo_wren_sig and clken;
data_fifo_wdata <= data_fifo_wdata_sig;
four_bit_proc : process (clock) -- make trigger sig a single clock width pulse
begin
	if rising_edge(clock) then
		if (four_bit_mode = '1') then
			clken <= not clken;
		else
			clken <= '1';
		end if;
	end if;
end process;
crc_err_proc : process (clock) -- register crc err after reset
begin
	if rising_edge(clock) then
		if (reset = '1') then
			crc_err_reg <= '0';
			crc_err_flag <= '0';
		elsif (user_crc_err = '1') then
		  	crc_err_reg <= '1';
		  	--stay high until reset to indicate there was an error ever
			crc_err_flag <= '1';
		elsif (clear_crc_err_flag = '1') then -- command handler clears err flag after reseting fifos
		   	crc_err_flag <= '0';
		end if;
	end if;
end process;

----------------------------------------------------------------------
-- Machine: Sreg0
----------------------------------------------------------------------
Sreg0_machine: process (clock)
begin
	if clock'event and clock = '1' then
		if reset = '1' then
			Sreg0 <= idle;
			-- Set default values for outputs, signals and variables
			data_fifo_wren_sig <= '0';
			info_fifo_wren_sig <= '0';
			info_fifo_wr_data(7 downto 0) <= (others=>'0');
			com_code <= '0';
			q_w_count <= (others=>'0');
			q_w_counter <= (others=>'0');
			data_fifo_wdata_sig <= (others=>'0');
		else
			if clken = '1' then
				-- Set default values for outputs, signals and variables
				data_fifo_wren_sig <= '0';
				info_fifo_wren_sig <= '0';
				case Sreg0 is
					when idle =>
						if user_rx_valid_out = '1' then
							Sreg0 <= S13;
							-- Put the entire command/size word in the info FIFO.
							-- This is not written until all info for the fifo is accumulated.
							info_fifo_wr_data(7 downto 0) <= user_rx_data_out;
							com_code <= user_rx_data_out(0);
							-- save command code
						end if;
					when insert_crc =>
						if user_crc_chk = '1' then	-- it's possible that there was an error in the protocol that was sent.. -- and so must wait for transimission to complete
							Sreg0 <= rcvdone;
						end if;
					when rcvdone =>
						Sreg0 <= idle;
						info_fifo_wr_data(7) <= crc_err_reg;
						-- return the crc error status in info, info complete
						info_fifo_wren_sig <= '1';
						-- we actually write the info fifo here.
					when S13 =>
						if (unsigned(user_rx_data_out) = 0)	-- size is 0
							and (user_rx_valid_out = '1') then	-- multi-operation packet
							Sreg0 <= idle;
							info_fifo_wr_data(15 downto 8) <= user_rx_data_out;
							info_fifo_wr_data(7) <= crc_err_reg;
							-- return the crc error status in info, info complete
							info_fifo_wren_sig <= '1';
							-- we actually write the info fifo here.
						elsif unsigned(user_rx_data_out) = 0 then	-- size is 0
							Sreg0 <= insert_crc;
							info_fifo_wr_data(15 downto 8) <= user_rx_data_out;
						else
							Sreg0 <= S3_S4;
							info_fifo_wr_data(15 downto 8) <= user_rx_data_out;
							q_w_counter <= (others => '0');
							-- initialize the counter
							-- Increment the quad word count for writes
							-- First word is the starting address
							-- next n words (I received a count of n)
							-- is the actual quad word data for writing.
							if (com_code = '1') then --write data coming
								q_w_count <= unsigned(user_rx_data_out) + 1;
							-- watch out for overflow
							-- writes the starting address first plus the
							-- data
							else -- all other commands only have 1
							    q_w_count <= (0=>'1',others => '0');
							-- This will cause only the address word
							-- to be written to the data fifo.
							end if;
						end if;
					when S3_S4 =>
						Sreg0 <= S3_S5;
						-- Finish the write to the data fifo
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
						q_w_counter <= q_w_counter + 1;
						-- increment the counter
					when S3_S9 =>
						Sreg0 <= S3_S10;
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
					when S3_S8 =>
						Sreg0 <= S3_S9;
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
					when S3_S7 =>
						Sreg0 <= S3_S8;
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
					when S3_S6 =>
						Sreg0 <= S3_S7;
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
					when S3_S5 =>
						Sreg0 <= S3_S6;
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
					when S3_S11 =>
						if (q_w_counter = q_w_count) and (user_rx_valid_out = '1') then	-- multi-operation packet
							Sreg0 <= idle;
							data_fifo_wren_sig <= '1';
							-- write the assembled data to the FIFO
							data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
							info_fifo_wr_data(7) <= crc_err_reg;
							-- return the crc error status in info, info complete
							info_fifo_wren_sig <= '1';
							-- we actually write the info fifo here.
						elsif q_w_counter = q_w_count then
							Sreg0 <= insert_crc;
							data_fifo_wren_sig <= '1';
							-- write the assembled data to the FIFO
							data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
						else
							Sreg0 <= S3_S4;
							data_fifo_wren_sig <= '1';
							-- write the assembled data to the FIFO
							data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
						end if;
					when S3_S10 =>
						Sreg0 <= S3_S11;
						data_fifo_wdata_sig <= user_rx_data_out & data_fifo_wdata_sig(63 downto 8);
--vhdl_cover_off
					when others =>
						null;
--vhdl_cover_on
				end case;
			end if;
		end if;
	end if;
end process;

end arch;
