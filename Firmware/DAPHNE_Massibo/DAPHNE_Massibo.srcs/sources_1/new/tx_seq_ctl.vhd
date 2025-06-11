-------------------------------------------------------------------------------
--
-- Title       : GEC_TX_SEQ_CTL
-- Design      : ethernet_controller
-- Author      : aprosser
-- Company     : CD_CEPA_ESE
--
-------------------------------------------------------------------------------
--
-- File        : d:\Projects\otsdaq\OtS Ethernet MAC firmware\ActiveHDL_proj\ethernet_controller\compile\tx_seq_ctl.vhd
-- Generated   : 02/29/16 11:06:13
-- From        : d:/Projects/otsdaq/OtS Ethernet MAC firmware/ActiveHDL_proj/ethernet_controller/src/tx_seq_ctl.asf
-- By          : FSM2VHDL ver. 5.0.7.2
--
-------------------------------------------------------------------------------
--
-- Description : 
--
-------------------------------------------------------------------------------

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.NUMERIC_STD.all;
use work.params_package.all;

entity tx_seq_ctl is 
	port (
		clk: in STD_LOGIC;
		ctrl_info_fifo_empty: in STD_LOGIC;
		data_fifo_empty: in STD_LOGIC;
		data_fifo_rd_data: in STD_LOGIC_VECTOR (63 downto 0);
		data_info_fifo_empty: in STD_LOGIC;
		dest_busy: in STD_LOGIC;
		four_bit_mode: in STD_LOGIC;
		info_fifo_rd_data: in STD_LOGIC_VECTOR (15 downto 0);
		reset: in STD_LOGIC;
		user_tx_enable_out: in STD_LOGIC;
		data_fifo_rden: out STD_LOGIC;
		fifo_sel: out STD_LOGIC;
		info_fifo_rden: out STD_LOGIC;
		ret_to_sender: out STD_LOGIC;
		tx_data: out STD_LOGIC_VECTOR (7 downto 0);
		user_trigger: out STD_LOGIC;
		user_tx_size_in: out STD_LOGIC_VECTOR (10 downto 0));
end tx_seq_ctl;

architecture arch of tx_seq_ctl is

-- diagram signals declarations
signal byte_count: UNSIGNED (2 downto 0);
signal clken: STD_LOGIC;
signal data_fifo_rd_data_reg: STD_LOGIC_VECTOR (63 downto 0);
signal data_fifo_rden_sig: STD_LOGIC;
signal fifo_sel_sig: STD_LOGIC;
signal info_fifo_rden_sig: STD_LOGIC;
signal qw_count: UNSIGNED (7 downto 0);
signal seq_count: UNSIGNED (7 downto 0);
signal tx_data_count: UNSIGNED (10 downto 0);

-- BINARY ENCODED state machine: Sreg0
attribute ENUM_ENCODING: string;
type Sreg0_type is (
    S7, S1, savecount, read_Ififo, S2, idle, txmtdone, chk_busy, S5, trgrd
);
attribute ENUM_ENCODING of Sreg0_type: type is
	"0000 " &		-- S7
	"0001 " &		-- S1
	"0010 " &		-- savecount
	"0011 " &		-- read_Ififo
	"0100 " &		-- S2
	"0101 " &		-- idle
	"0110 " &		-- txmtdone
	"0111 " &		-- chk_busy
	"1000 " &		-- S5
	"1001" ;		-- trgrd

signal Sreg0: Sreg0_type;

attribute STATE_VECTOR: string;
attribute STATE_VECTOR of arch: architecture is "Sreg0";

begin

-- concurrent signals assignments

-- Diagram ACTION
info_fifo_rden <= info_fifo_rden_sig and clken;
data_fifo_rden <= data_fifo_rden_sig and clken;
fifo_sel <= fifo_sel_sig;
four_bit_proc : process (clk)
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
			info_fifo_rden_sig <= '0';
			fifo_sel_sig <= '0';
			data_fifo_rden_sig <= '0';
			seq_count <= (others => '0');
			user_trigger <= '0';
			user_tx_size_in <= (others => '0');
			qw_count <= (others => '0');
			tx_data_count <= (others => '0');
			data_fifo_rd_data_reg <= (others => '0');
			byte_count <= (others => '0');
			fifo_sel_sig <= '0';
			ret_to_sender <= '0';
		else
			if clken = '1' then
				-- Set default values for outputs, signals and variables
				info_fifo_rden_sig <= '0';
				fifo_sel_sig <= '0';
				data_fifo_rden_sig <= '0';
				case Sreg0 is
					when S7 =>
						Sreg0 <= txmtdone;
						user_trigger <= '0';
						seq_count <= seq_count + 1;
					when S1 =>
						Sreg0 <= S2;
						-- add return code and sequence counter byte to
						-- produce the final number of bytes
						tx_data_count <= tx_data_count + 2;
					when savecount =>
						Sreg0 <= S1;
						-- get number of quad words
						qw_count <= unsigned(info_fifo_rd_data(15 downto 8));
						-- assert the return code to the Ethernet Controller
						-- so that the first byte is ready and waiting
						tx_data(7 downto 2) <= info_fifo_rd_data(7 downto 2);
						if(fifo_sel_sig = '1') then -- data fifo, use burst codes
							tx_data(1 downto 0) <= info_fifo_rd_data(1 downto 0);
							ret_to_sender <= '0';
						else -- ctrl fifo, use ret to sender bit
							tx_data(1 downto 0) <= '0' & info_fifo_rd_data(0);
							ret_to_sender <= not info_fifo_rd_data(1);
						end if;
						-- compute number of bytes in quad words to be returned to PC
						-- multiplies quad word count by 8
						tx_data_count <= unsigned(info_fifo_rd_data(15 downto 8) & "000");
						if (unsigned(info_fifo_rd_data(15 downto 8)) /= 0) then
						-- read a data quad word for initialization
						  	data_fifo_rden_sig <= '1';
						end if;
					when read_Ififo =>
						Sreg0 <= savecount;
					when S2 =>
						Sreg0 <= chk_busy;
						user_tx_size_in <= std_logic_vector(tx_data_count);
						-- present byte count to GEC
					when idle =>
						if data_info_fifo_empty = '0'  or
							ctrl_info_fifo_empty = '0' then	-- Then there is data to send!
							Sreg0 <= read_Ififo;
							-- Read the info word
							info_fifo_rden_sig <= '1';
							byte_count <= (others => '0');
							-- 0 for ctrl fifo, 1 for burst data fifo
							fifo_sel_sig <= not data_info_fifo_empty;
						end if;
					when txmtdone =>
						Sreg0 <= idle;
					when chk_busy =>
						if dest_busy = '0' then
							Sreg0 <= trgrd;
							-- EC not busy, assert trigger
							user_trigger <= '1';
							-- prepare first quad word if there is one
							data_fifo_rd_data_reg <= data_fifo_rd_data;
						elsif dest_busy = '1' then
							Sreg0 <= chk_busy;
						end if;
					when S5 =>
						if byte_count = 0 and qw_count = 0 then
							Sreg0 <= S7;
						else
							Sreg0 <= S5;
							tx_data <= data_fifo_rd_data_reg(7 downto 0);
							byte_count <= byte_count + 1;
							if (byte_count = 2 and qw_count > 1)then
							-- Read a new data quad word from data fifo with plenty of time to spare
								data_fifo_rden_sig <= '1';
							end if;
							if (byte_count = 7) then	--ready for next quadword
							-- register next quad word from data fifo
								data_fifo_rd_data_reg <= data_fifo_rd_data;
							    qw_count <= qw_count - 1;
							else 	-- on current quadword
							-- shift next byte into position
								data_fifo_rd_data_reg <= x"00" & data_fifo_rd_data_reg(63 downto 8);
							end if;
						end if;
					when trgrd =>
						if user_tx_enable_out = '1' then
							Sreg0 <= S5;
							tx_data <= std_logic_vector(seq_count);
						elsif -- trigger sent, wait for tx enable
							user_tx_enable_out = '0' then
							Sreg0 <= trgrd;
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

end arch;
