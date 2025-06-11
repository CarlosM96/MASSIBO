-- daphne_package.vhd
-- Jamieson Olsen <jamieson@fnal.gov>

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package daphne_package is

    -- Set static ethernet addresses here. MAC = 00:80:55:EC:00:0C and IP = 192.168.0.200

    constant OEI_NETWORK_NAME : std_logic_vector(23 downto 0) := x"C0A800"; --192.168.0.X;
    constant OEI_USR_ADDR     : std_logic_vector(7 downto 0)  := X"C8";     -- 192.168.0.200

    -- Address Mapping using the std_match notation '-' is a "don't care" bit

    constant BRAM0_ADDR    : std_logic_vector(31 downto 0) := "0000000000000111000000----------";  -- 0x00070000-0x000703FF
    constant DEADBEEF_ADDR : std_logic_vector(31 downto 0) := X"0000aa55";
    constant STATVEC_ADDR  : std_logic_vector(31 downto 0) := X"00001974";
    constant GITVER_ADDR   : std_logic_vector(31 downto 0) := X"00009000";
    constant TESTREG_ADDR  : std_logic_vector(31 downto 0) := X"12345678";
    
    
    constant FIFO_TOP_ADDR     : std_logic_vector(31 downto 0) := X"80000000";
    constant FIFO_MID_ADDR     : std_logic_vector(31 downto 0) := X"90000000";
    constant FIFO_BOT_ADDR     : std_logic_vector(31 downto 0) := X"A0000000";
    
    constant FIFO_TOP_WR_ADDR     : std_logic_vector(31 downto 0) := X"8000000A";
    constant FIFO_MID_WR_ADDR     : std_logic_vector(31 downto 0) := X"9000000A";
    constant FIFO_BOT_WR_ADDR     : std_logic_vector(31 downto 0) := X"A000000A";
    
    constant FIFO_TOP_TS_ADDR     : std_logic_vector(31 downto 0) := X"80000010";
    constant FIFO_MID_TS_ADDR     : std_logic_vector(31 downto 0) := X"90000010";
    constant FIFO_BOT_TS_ADDR     : std_logic_vector(31 downto 0) := X"A0000010";
    
   constant FIFO_TOP_ERR_ADDR     : std_logic_vector(31 downto 0) := X"8000000B";
   constant FIFO_MID_ERR_ADDR     : std_logic_vector(31 downto 0) := X"9000000B";
   constant FIFO_BOT_ERR_ADDR     : std_logic_vector(31 downto 0) := X"A000000B";  
   
   constant FIFO_READABLE     : std_logic_vector(31 downto 0) := X"B000000B";  
   
   
   
   constant TOP_THRESHOLD     : std_logic_vector(31 downto 0) := X"00006008";
   constant MID_THRESHOLD     : std_logic_vector(31 downto 0) := X"00006009";
   constant BOT_THRESHOLD     : std_logic_vector(31 downto 0) := X"0000600A";  
   
   
   
    type array_3x14_type is array (2 downto 0) of std_logic_vector(13 downto 0);
    type array_5x9_type is array (4 downto 0) of std_logic_vector(8 downto 0);
    type array_9x16_type is array (8 downto 0) of std_logic_vector(15 downto 0);
    type array_5x9x16_type is array (4 downto 0) of array_9x16_type;

    -- DAPHNE specific addresses

    -- AFE 
    constant AFE : integer := 0;
    -- write anything to this address to force trigger

    constant TRIGGER_ADDR               : std_logic_vector(31 downto 0) := X"00002000";
    constant TRIGGER_ADDR_2               : std_logic_vector(31 downto 0) := X"00002012";
    constant TRIGGER_INT_DEAD_TIME_ADDR : std_logic_vector(31 downto 0) := X"00002020";

    -- write anything to this address to force reset of the AFE front end logic

    constant RESETFE_ADDR : std_logic_vector(31 downto 0) := X"00002001";

    -- write anything to bitslip the specified channel, note these are not separate

    constant BITSLIP_AFE0_D0_ADDR : std_logic_vector(31 downto 0) := X"00003000";
    constant BITSLIP_AFE0_D1_ADDR : std_logic_vector(31 downto 0) := X"00003001";
    constant BITSLIP_AFE0_D2_ADDR : std_logic_vector(31 downto 0) := X"00003002";
    constant BITSLIP_AFE0_D3_ADDR : std_logic_vector(31 downto 0) := X"00003003";
    constant BITSLIP_AFE0_D4_ADDR : std_logic_vector(31 downto 0) := X"00003004";
    constant BITSLIP_AFE0_D5_ADDR : std_logic_vector(31 downto 0) := X"00003005";
    constant BITSLIP_AFE0_D6_ADDR : std_logic_vector(31 downto 0) := X"00003006";
    constant BITSLIP_AFE0_D7_ADDR : std_logic_vector(31 downto 0) := X"00003007";
    constant BITSLIP_AFE0_FR_ADDR : std_logic_vector(31 downto 0) := X"00003008";

    constant BITSLIP_AFE1_D0_ADDR : std_logic_vector(31 downto 0) := X"00003010";
    constant BITSLIP_AFE1_D1_ADDR : std_logic_vector(31 downto 0) := X"00003011";
    constant BITSLIP_AFE1_D2_ADDR : std_logic_vector(31 downto 0) := X"00003012";
    constant BITSLIP_AFE1_D3_ADDR : std_logic_vector(31 downto 0) := X"00003013";
    constant BITSLIP_AFE1_D4_ADDR : std_logic_vector(31 downto 0) := X"00003014";
    constant BITSLIP_AFE1_D5_ADDR : std_logic_vector(31 downto 0) := X"00003015";
    constant BITSLIP_AFE1_D6_ADDR : std_logic_vector(31 downto 0) := X"00003016";
    constant BITSLIP_AFE1_D7_ADDR : std_logic_vector(31 downto 0) := X"00003017";
    constant BITSLIP_AFE1_FR_ADDR : std_logic_vector(31 downto 0) := X"00003018";

    constant BITSLIP_AFE2_D0_ADDR : std_logic_vector(31 downto 0) := X"00003020";
    constant BITSLIP_AFE2_D1_ADDR : std_logic_vector(31 downto 0) := X"00003021";
    constant BITSLIP_AFE2_D2_ADDR : std_logic_vector(31 downto 0) := X"00003022";
    constant BITSLIP_AFE2_D3_ADDR : std_logic_vector(31 downto 0) := X"00003023";
    constant BITSLIP_AFE2_D4_ADDR : std_logic_vector(31 downto 0) := X"00003024";
    constant BITSLIP_AFE2_D5_ADDR : std_logic_vector(31 downto 0) := X"00003025";
    constant BITSLIP_AFE2_D6_ADDR : std_logic_vector(31 downto 0) := X"00003026";
    constant BITSLIP_AFE2_D7_ADDR : std_logic_vector(31 downto 0) := X"00003027";
    constant BITSLIP_AFE2_FR_ADDR : std_logic_vector(31 downto 0) := X"00003028";

    constant BITSLIP_AFE3_D0_ADDR : std_logic_vector(31 downto 0) := X"00003030";
    constant BITSLIP_AFE3_D1_ADDR : std_logic_vector(31 downto 0) := X"00003031";
    constant BITSLIP_AFE3_D2_ADDR : std_logic_vector(31 downto 0) := X"00003032";
    constant BITSLIP_AFE3_D3_ADDR : std_logic_vector(31 downto 0) := X"00003033";
    constant BITSLIP_AFE3_D4_ADDR : std_logic_vector(31 downto 0) := X"00003034";
    constant BITSLIP_AFE3_D5_ADDR : std_logic_vector(31 downto 0) := X"00003035";
    constant BITSLIP_AFE3_D6_ADDR : std_logic_vector(31 downto 0) := X"00003036";
    constant BITSLIP_AFE3_D7_ADDR : std_logic_vector(31 downto 0) := X"00003037";
    constant BITSLIP_AFE3_FR_ADDR : std_logic_vector(31 downto 0) := X"00003038";

    constant BITSLIP_AFE4_D0_ADDR : std_logic_vector(31 downto 0) := X"00003040";
    constant BITSLIP_AFE4_D1_ADDR : std_logic_vector(31 downto 0) := X"00003041";
    constant BITSLIP_AFE4_D2_ADDR : std_logic_vector(31 downto 0) := X"00003042";
    constant BITSLIP_AFE4_D3_ADDR : std_logic_vector(31 downto 0) := X"00003043";
    constant BITSLIP_AFE4_D4_ADDR : std_logic_vector(31 downto 0) := X"00003044";
    constant BITSLIP_AFE4_D5_ADDR : std_logic_vector(31 downto 0) := X"00003045";
    constant BITSLIP_AFE4_D6_ADDR : std_logic_vector(31 downto 0) := X"00003046";
    constant BITSLIP_AFE4_D7_ADDR : std_logic_vector(31 downto 0) := X"00003047";
    constant BITSLIP_AFE4_FR_ADDR : std_logic_vector(31 downto 0) := X"00003048";

    -- write idelay value (range 0-31) to all channels of the AFE chip

    constant DELAY_AFE0_ADDR : std_logic_vector(31 downto 0) := X"00004000";
    constant DELAY_AFE1_ADDR : std_logic_vector(31 downto 0) := X"00004001";
    constant DELAY_AFE2_ADDR : std_logic_vector(31 downto 0) := X"00004002";
    constant DELAY_AFE3_ADDR : std_logic_vector(31 downto 0) := X"00004003";
    constant DELAY_AFE4_ADDR : std_logic_vector(31 downto 0) := X"00004004";
    
    -- THRESHOLD     
    constant THRESHOLD_BASEADDR: std_logic_vector(31 downto 0) := X"00006000";
    constant RESET_THRESHOLD: std_logic_vector(31 downto 0) := X"00006001";
    constant DEFAULT_THRESHOLD: std_logic_vector(13 downto 0) := "00000100000000";
    
    
    constant SELF_TRIGGER_MODE: std_logic_vector(31 downto 0) := X"00006010";
    constant SOFT_TRIGGER_MODE: std_logic_vector(31 downto 0) := X"00006015";
    constant EXT_TRIGGER_MODE: std_logic_vector(31 downto 0) := X"0000610A";
    

    -- spy buffers are 4k deep

    constant SPYBUF_AFE0_D0_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000000----------------";
    constant SPYBUF_AFE0_D1_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000001----------------";
    constant SPYBUF_AFE0_D2_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000010----------------";
    constant SPYBUF_AFE0_D3_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000011----------------";
    constant SPYBUF_AFE0_D4_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000100----------------";
    constant SPYBUF_AFE0_D5_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000101----------------";
    constant SPYBUF_AFE0_D6_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000110----------------";
    constant SPYBUF_AFE0_D7_BASEADDR : std_logic_vector(31 downto 0) := "0100000000000111----------------";
    constant SPYBUF_AFE0_FR_BASEADDR : std_logic_vector(31 downto 0) := "0100000000001000----------------";

    constant SPYBUF_AFE1_D0_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010000----------------";
    constant SPYBUF_AFE1_D1_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010001----------------";
    constant SPYBUF_AFE1_D2_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010010----------------";
    constant SPYBUF_AFE1_D3_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010011----------------";
    constant SPYBUF_AFE1_D4_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010100----------------";
    constant SPYBUF_AFE1_D5_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010101----------------";
    constant SPYBUF_AFE1_D6_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010110----------------";
    constant SPYBUF_AFE1_D7_BASEADDR : std_logic_vector(31 downto 0) := "0100000000010111----------------";
    constant SPYBUF_AFE1_FR_BASEADDR : std_logic_vector(31 downto 0) := "0100000000011000----------------";

    constant SPYBUF_AFE2_D0_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100000----------------";
    constant SPYBUF_AFE2_D1_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100001----------------";
    constant SPYBUF_AFE2_D2_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100010----------------";
    constant SPYBUF_AFE2_D3_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100011----------------";
    constant SPYBUF_AFE2_D4_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100100----------------";
    constant SPYBUF_AFE2_D5_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100101----------------";
    constant SPYBUF_AFE2_D6_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100110----------------";
    constant SPYBUF_AFE2_D7_BASEADDR : std_logic_vector(31 downto 0) := "0100000000100111----------------";
    constant SPYBUF_AFE2_FR_BASEADDR : std_logic_vector(31 downto 0) := "0100000000101000----------------";

    constant SPYBUF_AFE3_D0_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110000----------------";
    constant SPYBUF_AFE3_D1_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110001----------------";
    constant SPYBUF_AFE3_D2_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110010----------------";
    constant SPYBUF_AFE3_D3_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110011----------------";
    constant SPYBUF_AFE3_D4_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110100----------------";
    constant SPYBUF_AFE3_D5_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110101----------------";
    constant SPYBUF_AFE3_D6_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110110----------------";
    constant SPYBUF_AFE3_D7_BASEADDR : std_logic_vector(31 downto 0) := "0100000000110111----------------";
    constant SPYBUF_AFE3_FR_BASEADDR : std_logic_vector(31 downto 0) := "0100000000111000----------------";

    constant SPYBUF_AFE4_D0_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000000----------------";
    constant SPYBUF_AFE4_D1_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000001----------------";
    constant SPYBUF_AFE4_D2_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000010----------------";
    constant SPYBUF_AFE4_D3_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000011----------------";
    constant SPYBUF_AFE4_D4_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000100----------------";
    constant SPYBUF_AFE4_D5_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000101----------------";
    constant SPYBUF_AFE4_D6_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000110----------------";
    constant SPYBUF_AFE4_D7_BASEADDR : std_logic_vector(31 downto 0) := "0100000001000111----------------";
    constant SPYBUF_AFE4_FR_BASEADDR : std_logic_vector(31 downto 0) := "0100000001001000----------------";

    -- spy buffer for the 64 bit timestamp value

    constant SPYBUFTS_BASEADDR : std_logic_vector(31 downto 0) := "0100000001010000----------------";

end package;
